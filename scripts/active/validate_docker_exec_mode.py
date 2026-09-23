"""Validate DOCKER_MODE=exec against the original per-test `docker run` mode.

Replays recorded test sequences (from an existing RepoExploreBench result
file) through both runner modes and compares, test by test, the new-branch
counts, the final cumulative coverage and pass/fail, plus wall time. Also
checks the exec-mode edge cases: a hanging test (timeout -> fresh container)
and a crashing test followed by a normal one.

Usage:
    python scripts/active/validate_docker_exec_mode.py \
        --input results/repo_explore_bench/full_run_gemini.json --per-repo 1
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner  # noqa: E402
from curiosity_explorer.benchmarks.repo_explore_bench import load_benchmark  # noqa: E402


def make_runner(target, mode):
    return DockerCoverageRunner(
        image=target["docker_image"], source_module=target["module"],
        setup_code=target["setup_code"], working_dir=target["working_dir"],
        env=target["env"], mode=mode)


def replay(target, scripts, mode):
    runner = make_runner(target, mode)
    t = time.time()
    per_test = []
    for sc in scripts:
        r = runner.run_test(sc)
        per_test.append((r.new_branches, r.passed, r.exception))
    elapsed = time.time() - t
    final = runner.get_cumulative_coverage()
    runner.cleanup()
    return {"per_test": per_test, "final": final, "elapsed": elapsed}


def compare(target, scripts):
    run = replay(target, scripts, "run")
    exe = replay(target, scripts, "exec")
    diffs = [i for i, (a, b) in enumerate(zip(run["per_test"], exe["per_test"]))
             if a[:2] != b[:2]]
    return target["module"], run, exe, diffs


def edge_cases(target):
    r = make_runner(target, "exec")
    out = {}
    res = r.run_test("while True: pass", timeout=5)
    out["hang -> TimeoutError"] = res.exception == "TimeoutError"
    out["container dropped after timeout"] = r._container is None
    res = r.run_test("raise ValueError('boom')")
    out["crash -> not passed"] = not res.passed
    res = r.run_test(f"import {target['module']}\nprint('ok')")
    out["normal test after crash passes"] = bool(res.passed)
    out["no stale coverage (import covers >0)"] = res.cumulative_branches > 0
    r.cleanup()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/repo_explore_bench/full_run_gemini.json")
    ap.add_argument("--strategy", default="cov_qvalue")
    ap.add_argument("--per-repo", type=int, default=1)
    args = ap.parse_args()

    data = json.load(open(ROOT / args.input))
    traces = {r["module"]: [t["script"] for t in r["strategies"][args.strategy]["trace"]]
              for r in data["results"] if args.strategy in r["strategies"]}
    seen, targets = {}, []
    for t in load_benchmark():
        if t["module"] in traces and seen.get(t["repo"], 0) < args.per_repo:
            seen[t["repo"]] = seen.get(t["repo"], 0) + 1
            targets.append(t)

    with ThreadPoolExecutor(len(targets)) as ex:
        rows = list(ex.map(lambda t: compare(t, traces[t["module"]]), targets))

    n_tests = n_diff = 0
    t_run = t_exec = 0.0
    print(f"{'module':<34}{'tests':>6}{'final run':>10}{'final exec':>11}"
          f"{'diff tests':>11}{'run s':>8}{'exec s':>8}")
    for module, run, exe, diffs in rows:
        n_tests += len(run["per_test"])
        n_diff += len(diffs)
        t_run += run["elapsed"]
        t_exec += exe["elapsed"]
        print(f"{module:<34}{len(run['per_test']):>6}{run['final']:>10}"
              f"{exe['final']:>11}{len(diffs):>11}{run['elapsed']:>8.0f}"
              f"{exe['elapsed']:>8.0f}")
        for i in diffs[:3]:
            print(f"    test {i}: run={run['per_test'][i]} exec={exe['per_test'][i]}")
    print(f"\n{n_diff}/{n_tests} tests differ; "
          f"finals equal on {sum(r[1]['final'] == r[2]['final'] for r in rows)}/{len(rows)} targets; "
          f"speedup {t_run / t_exec:.1f}x ({t_run:.0f}s -> {t_exec:.0f}s)")

    print("\nexec-mode edge cases:")
    for k, v in edge_cases(targets[0]).items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    main()
