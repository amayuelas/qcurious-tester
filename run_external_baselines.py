"""External-baseline runner for Experiment 4 (rebuttal, COLM 2026).

Addresses sHcP/Cp1s/ovhA's request for comparison against published prior work.
Runs external test-generation tools on the SAME RepoExploreBench targets our
strategies use, then measures the branch coverage of each tool's generated
suite with the SAME coverage.py invocation DockerCoverageRunner uses
(`coverage run --branch --source=<module>`), so the resulting "final branches"
number is directly comparable to cov_qvalue / divhints_random / etc.

Baselines:
  - pynguin  : classical search-based test generation (non-LLM reference point).
               Native budget is search-time (seconds), not N executions; we
               report the search-time used (see --search-time) since Pynguin
               has no directly-comparable N=24-executions knob.
  - coverup  : coverage-guided iterative LLM test generation (added separately).

Generation + measurement happen in one container run on the
`curiositybench-baselines:latest` image (built from Dockerfile.baselines) so the
generated suite is exercised against the exact installed package versions.

Usage:
    python run_external_baselines.py --baseline pynguin --max-targets 1
    python run_external_baselines.py --baseline pynguin --repos click flask \
        --search-time 60 --output repo_explore_bench/exp4_pynguin.json
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.benchmarks.repo_explore_bench import (
    load_benchmark, get_benchmark_info,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASELINE_IMAGE = "curiositybench-baselines:latest"

# Separators printed inside the container to delimit phases on stdout.
SEP_GEN = "===GEN_DONE==="
SEP_TEST = "===PYTEST_DONE==="
SEP_COV = "===COV_JSON==="


def count_coverage(cov_json_str, target_file=None):
    """Count unique branches/lines from a coverage.py JSON blob.

    Mirrors DockerCoverageRunner's extraction exactly so the branch metric is
    identical to our strategies: branches keyed by (file, tuple(arc)); when a
    file reports no branches, fall back to executed lines as a branch proxy.
    """
    branches, lines = set(), set()
    if not cov_json_str:
        return branches, lines
    try:
        cov_data = json.loads(cov_json_str)
    except (json.JSONDecodeError, ValueError):
        return branches, lines
    for file_path, file_data in cov_data.get("files", {}).items():
        if target_file and target_file not in file_path:
            continue
        exec_branches = file_data.get("executed_branches", [])
        for arc in exec_branches:
            branches.add((file_path, tuple(arc)))
        for line in file_data.get("executed_lines", []):
            lines.add((file_path, line))
        if not exec_branches:
            for line in file_data.get("executed_lines", []):
                branches.add((file_path, line))
    return branches, lines


def run_pynguin_target(target, search_time, docker_timeout):
    """Generate a Pynguin suite for one target and measure its branch coverage.

    Returns a result dict with final branches/lines and diagnostics.
    """
    module = target["module"]
    image = BASELINE_IMAGE

    # One container run: locate site-packages, generate into /tmp/gen, then
    # measure coverage of the generated suite with the same coverage.py call.
    inner = (
        "set -o pipefail; "
        "PP=$(python -c \"import sysconfig;print(sysconfig.get_paths()['purelib'])\"); "
        "rm -rf /tmp/gen && mkdir -p /tmp/gen; "
        f"timeout {search_time + 120} pynguin --project_path \"$PP\" "
        f"--module-name {module} --output-path /tmp/gen "
        f"--maximum_search_time {search_time} "
        "--assertion-generation NONE -v 2>&1 | tail -25; "
        f"echo '{SEP_GEN}'; "
        f"coverage run --rcfile=/dev/null --branch --source={module} "
        "-m pytest /tmp/gen -q -p no:cacheprovider 2>&1 | tail -15; "
        f"echo '{SEP_TEST}'; "
        "coverage json --rcfile=/dev/null -o /tmp/cov.json 2>/dev/null && "
        f"echo '{SEP_COV}' && cat /tmp/cov.json 2>/dev/null"
    )
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "bash",
        "-e", "PYNGUIN_DANGER_AWARE=1",
        image, "-c", inner,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=docker_timeout)
        raw = proc.stdout
    except subprocess.TimeoutExpired:
        return {"module": module, "repo": target["repo"], "error": "docker_timeout",
                "final": 0, "final_lines": 0, "elapsed": time.time() - t0}
    except Exception as e:
        return {"module": module, "repo": target["repo"], "error": str(e),
                "final": 0, "final_lines": 0, "elapsed": time.time() - t0}

    gen_log = test_log = cov_json = ""
    if SEP_GEN in raw:
        gen_log, rest = raw.split(SEP_GEN, 1)
    else:
        rest = raw
    if SEP_TEST in rest:
        test_log, rest = rest.split(SEP_TEST, 1)
    if SEP_COV in rest:
        cov_json = rest.split(SEP_COV, 1)[1].strip()

    branches, lines = count_coverage(cov_json)

    # Diagnostics: pytest collected/passed summary + any generation failure.
    summary = ""
    for ln in (test_log or "").strip().splitlines()[-3:]:
        if "passed" in ln or "error" in ln or "collected" in ln or "no tests" in ln:
            summary = ln.strip()
    return {
        "module": module,
        "repo": target["repo"],
        "final": len(branches),
        "final_lines": len(lines),
        "pytest_summary": summary,
        "gen_tail": (gen_log or "").strip()[-300:],
        "elapsed": round(time.time() - t0, 1),
    }


def run_coverup_target(target, model, docker_timeout):
    """Generate a CoverUp suite for one target and measure its branch coverage.

    CoverUp is coverage-guided iterative LLM test generation. We point it at the
    single module's source file (positional arg) within its package dir, route
    its litellm calls to gemini via the OpenAI-compatible endpoint, then measure
    the generated suite's coverage with the same coverage.py call as Pynguin.

    NOTE: validated against CoverUp's CLI but not yet smoke-run end-to-end
    (the gemini API was occupied by the Exp-1 rerun). Run one target first.
    """
    module = target["module"]
    top = module.split(".")[0]
    src_rel = module.replace(".", "/") + ".py"  # e.g. click/core.py

    # CoverUp drives the LLM through litellm (its own dependency), and litellm
    # gatekeeps function calling on a static model-name registry. For a local
    # vLLM model, use litellm's purpose-built `hosted_vllm/` provider, which
    # passes tool calls straight through to the vLLM server (which must be
    # started with --enable-auto-tool-choice --tool-call-parser gemma4). The
    # container reaches the host's server via --network host. For an external
    # API model (gemini) fall back to the OpenAI-compatible endpoint.
    is_vllm = model.startswith("google/") or "gemma" in model.lower()
    if is_vllm:
        litellm_model = f"hosted_vllm/{model}"
        # Mount sitecustomize.py that registers the model as tool-capable in
        # litellm (its static registry otherwise refuses function calling for
        # the custom served-model name, before reaching vLLM).
        inject_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "scripts", "coverup_inject")
        env = ["-e", f"HOSTED_VLLM_API_BASE={config.VLLM_API_BASE}",
               "-e", f"HOSTED_VLLM_API_KEY={config.VLLM_API_KEY}",
               "-e", "PYTHONPATH=/inject",
               "-e", f"COVERUP_VLLM_MODEL={model}"]
        net = ["--network", "host", "-v", f"{inject_dir}:/inject:ro"]
    else:
        litellm_model = f"openai/{model}"
        env = ["-e", f"OPENAI_API_BASE={config.GEMINI_API_BASE}",
               "-e", f"OPENAI_API_KEY={config.GEMINI_API_KEY}"]
        net = []

    # CoverUp's per-candidate coverage check (testrunner.measure_test_coverage)
    # runs slipcover WITHOUT --source, so slipcover only instruments the working
    # tree, not site-packages. Our targets are pip-installed, so CoverUp would
    # measure 0% and discard every test. Fix: generate against a LOCAL COPY of
    # the package (so it's in slipcover's default scope), but measure OUR
    # comparable metric against the installed package (same coverage.py
    # --source call as Pynguin/our strategies). PYTHONPATH is cleared for the
    # path-detection + measurement calls so the injected litellm sitecustomize
    # banner doesn't pollute them.
    inner = (
        "set -o pipefail; "
        "PP=$(PYTHONPATH= python -c \"import sysconfig;print(sysconfig.get_paths()['purelib'])\"); "
        "rm -rf /tmp/gen /work && mkdir -p /tmp/gen /work; "
        f"cp -r \"$PP/{top}\" /work/{top}; cd /work; "
        f"coverup --package-dir {top} --tests-dir /tmp/gen "
        f"--model {litellm_model} --model-temperature 0 --no-checkpoint "
        f"--no-isolate-tests {src_rel} 2>&1 | grep -vE 'LiteLLM|botocore' | tail -25; "
        f"echo '{SEP_GEN}'; "
        "cd \"$PP\"; "
        f"PYTHONPATH= coverage run --rcfile=/dev/null --branch --source={module} "
        "-m pytest /tmp/gen -q -p no:cacheprovider 2>&1 | tail -15; "
        f"echo '{SEP_TEST}'; "
        "PYTHONPATH= coverage json --rcfile=/dev/null -o /tmp/cov.json 2>/dev/null && "
        f"echo '{SEP_COV}' && cat /tmp/cov.json 2>/dev/null"
    )
    cmd = [
        "docker", "run", "--rm", "--entrypoint", "bash", *net, *env,
        BASELINE_IMAGE, "-c", inner,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=docker_timeout)
        raw = proc.stdout
    except subprocess.TimeoutExpired:
        return {"module": module, "repo": target["repo"], "error": "docker_timeout",
                "final": 0, "final_lines": 0, "elapsed": time.time() - t0}
    except Exception as e:
        return {"module": module, "repo": target["repo"], "error": str(e),
                "final": 0, "final_lines": 0, "elapsed": time.time() - t0}

    gen_log = test_log = cov_json = ""
    rest = raw.split(SEP_GEN, 1)[1] if SEP_GEN in raw else raw
    gen_log = raw.split(SEP_GEN, 1)[0] if SEP_GEN in raw else ""
    if SEP_TEST in rest:
        test_log, rest = rest.split(SEP_TEST, 1)
    if SEP_COV in rest:
        cov_json = rest.split(SEP_COV, 1)[1].strip()

    branches, lines = count_coverage(cov_json)
    summary = ""
    for ln in (test_log or "").strip().splitlines()[-3:]:
        if any(k in ln for k in ("passed", "error", "collected", "no tests")):
            summary = ln.strip()
    return {
        "module": module, "repo": target["repo"],
        "final": len(branches), "final_lines": len(lines),
        "pytest_summary": summary,
        "gen_tail": (gen_log or "").strip()[-300:],
        "elapsed": round(time.time() - t0, 1),
    }


def parse_args():
    p = argparse.ArgumentParser(description="External-baseline runner (Exp 4)")
    p.add_argument("--baseline", choices=["pynguin", "coverup"],
                   default="pynguin")
    p.add_argument("--repos", nargs="+", default=None)
    p.add_argument("--modules", nargs="+", default=None,
                   help="Run only these exact module names (e.g. for re-running "
                        "timed-out targets at a higher --docker-timeout).")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--search-time", type=int, default=60,
                   help="Pynguin search-time budget in seconds (native budget).")
    p.add_argument("--model", default="gemini-3-flash-preview",
                   help="LLM for coverup (routed via litellm openai/ provider).")
    p.add_argument("--parallel", type=int, default=2,
                   help="Targets in parallel (kept low to share Docker with "
                        "other running jobs).")
    p.add_argument("--docker-timeout", type=int, default=None,
                   help="Per-target docker timeout (default: search-time + 240).")
    p.add_argument("--output", default="repo_explore_bench/exp4_pynguin.json")
    return p.parse_args()


def main():
    args = parse_args()
    # CoverUp does many LLM round-trips per module; give it a wider default.
    default_timeout = 900 if args.baseline == "coverup" else args.search_time + 240
    docker_timeout = args.docker_timeout or default_timeout
    targets = load_benchmark(repos=args.repos, max_targets=args.max_targets)
    if args.modules:
        wanted = set(args.modules)
        targets = [t for t in targets if t["module"] in wanted]

    print("=" * 70, flush=True)
    print(f"Exp 4 — external baseline: {args.baseline}", flush=True)
    print(f"  Targets: {len(targets)}  search-time: {args.search_time}s  "
          f"parallel: {args.parallel}", flush=True)
    print("=" * 70, flush=True)

    start = time.time()
    results = []
    done = [0]

    def run_one(target):
        if args.baseline == "pynguin":
            r = run_pynguin_target(target, args.search_time, docker_timeout)
        elif args.baseline == "coverup":
            r = run_coverup_target(target, args.model, docker_timeout)
        else:
            raise ValueError(args.baseline)
        done[0] += 1
        print(f"  [{done[0]}/{len(targets)}] {r['module']:<32} "
              f"branches={r['final']:<4} lines={r['final_lines']:<5} "
              f"{r.get('pytest_summary') or r.get('error', '')}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_one, t): t for t in targets}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                t = futures[f]
                print(f"  ERROR {t['module']}: {e}", flush=True)

    elapsed = time.time() - start

    finals = [r["final"] for r in results]
    print(f"\n{'=' * 70}\nRESULTS ({args.baseline})\n{'=' * 70}", flush=True)
    print(f"  mean branches: {statistics.mean(finals):.1f}  "
          f"(n={len(finals)})" if finals else "  no results", flush=True)
    by_repo = {}
    for r in results:
        by_repo.setdefault(r["repo"], []).append(r["final"])
    for repo in sorted(by_repo):
        vals = by_repo[repo]
        print(f"    {repo:<12} mean={statistics.mean(vals):.1f} (n={len(vals)})",
              flush=True)
    print(f"\n  Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)

    # Save in a shape that mirrors run_repo_explore_bench (per-target strategy
    # dict) so results can be merged into the comparison tables.
    out_results = [{
        "module": r["module"], "repo": r["repo"],
        "strategies": {args.baseline: {"final": r["final"],
                                       "final_lines": r["final_lines"]}},
        "diagnostics": {k: r.get(k) for k in
                        ("pytest_summary", "gen_tail", "error", "elapsed")},
    } for r in results]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = config.RESULTS_DIR / args.output
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": get_benchmark_info(),
            "config": {"baseline": args.baseline,
                       "search_time": args.search_time,
                       "image": BASELINE_IMAGE},
            "results": out_results,
            "elapsed": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
