"""Total number of branches per benchmark target, for coverage percentages.

Runs one import-only test per target with the same DockerCoverageRunner (and
the same --source / target-file filtering) the experiments use, and records
the module's static branch extent: executed + missing branch arcs reported by
coverage.py. Covered branches in the result files are counted from the same
arcs, so covered / total is the per-target branch coverage percentage. No LLM
calls.

When coverage.py reports no branch arcs for a target, the runner counts
executed lines as a branch proxy; the total then falls back to executed +
missing lines, matching that proxy.

Usage (from the repo root):
    python scripts/active/compute_branch_totals.py --benchmark reb
    python scripts/active/compute_branch_totals.py --benchmark tge --repos django/django
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner  # noqa: E402


def total_for(runner, module):
    try:
        runner.run_test(f"import {module}\n", timeout=120)
        branches = len(runner.all_branches)
        return branches if branches else len(runner.all_lines)
    finally:
        runner.cleanup()


def reb_jobs():
    from curiosity_explorer.benchmarks.repo_explore_bench import load_benchmark
    for t in load_benchmark():
        yield t["module"], t["module"], lambda t=t: DockerCoverageRunner(
            image=t["docker_image"], source_module=t["module"],
            setup_code=t["setup_code"], working_dir=t["working_dir"], env=t["env"])


def tge_jobs(repos):
    from curiosity_explorer.benchmarks.testgeneval_config import load_testgeneval_examples
    for ex in load_testgeneval_examples(repos=repos):
        parts = ex["module"].split(".")
        source = (".".join(parts[:-1]) if len(parts) >= 3
                  else parts[0] if len(parts) == 2 else ex["module"])
        key = f'{ex["code_file"]}@{ex.get("version")}'
        yield key, ex["module"], lambda ex=ex, source=source: DockerCoverageRunner(
            image=ex["image"], source_module=source, setup_code=ex["setup_code"],
            working_dir=ex["working_dir"], env=ex["env"],
            python_bin=ex.get("python_bin", "python"),
            pre_command=ex.get("pre_install") or "", target_file=ex.get("code_file"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["reb", "tge"], required=True)
    ap.add_argument("--repos", nargs="+", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    out = ROOT / "results" / f"branch_totals_{args.benchmark}.json"
    totals = json.load(open(out)) if out.exists() else {}
    jobs = [j for j in (reb_jobs() if args.benchmark == "reb" else tge_jobs(args.repos))
            if j[0] not in totals]
    print(f"{len(jobs)} targets to measure ({len(totals)} already in {out.name})")

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(total_for, make(), module): key for key, module, make in jobs}
        for f in as_completed(futs):
            key = futs[f]
            try:
                totals[key] = f.result()
            except Exception as e:  # keep going; report at the end
                print(f"  FAILED {key}: {e}")
                continue
            print(f"  {key}: {totals[key]}")
            json.dump(totals, open(out, "w"), indent=1, sort_keys=True)
    print(f"saved {len(totals)} totals -> {out}")


if __name__ == "__main__":
    main()
