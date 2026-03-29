"""Ablation: S (plan length) with matched rounds.

Each S value gets exactly 8 rounds of planning, so:
  S=1: 8 rounds × 1 script = 8 executions
  S=3: 8 rounds × 3 scripts = 24 executions
  S=5: 8 rounds × 5 scripts = 40 executions

This equalizes coverage map updates (8 per S value).
The question: does spending more executions per round on longer
plans discover more than spending fewer executions per round?
"""

import argparse
import random as _random
import re
import time
import json
import logging
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, generate_coverage_qvalue, _parse_script,
)
from curiosity_explorer.benchmarks.repo_explore_bench import (
    load_benchmark, DOCKER_IMAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

K = 3
GAMMA = 0.5
N_ROUNDS = 8
S_VALUES = [1, 3, 5]


def fetch_source(module_name):
    cmd = (f"docker run --rm {DOCKER_IMAGE} python3 -c "
           f"\"import inspect, {module_name}; print(inspect.getsource({module_name}))\"")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def run_one(target, source, S):
    """Run cov_qvalue with plan_length=S for N_ROUNDS rounds."""
    _random.seed(42)
    module = target["module"]
    exec_budget = N_ROUNDS * S

    runner = DockerCoverageRunner(
        image=target["docker_image"],
        source_module=module,
        setup_code=target["setup_code"],
        working_dir=target["working_dir"],
        env=target["env"],
    )

    hist = []
    cov_map = CoverageMap()
    executions = 0
    branch_curve = []

    while executions < exec_budget:
        scripts = generate_coverage_qvalue(source, module, hist, cov_map,
                                            K=K, plan_length=S, gamma=GAMMA)
        if not scripts:
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            continue

        for script in scripts:
            if executions >= exec_budget:
                break
            result = runner.run_test(script)
            hist.append((script, result))
            cov_map.update(script, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())

    return {
        "final": runner.get_cumulative_coverage(),
        "executions": executions,
        "branch_curve": branch_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    reset_cost()
    targets = load_benchmark(max_targets=args.max_targets)

    print(f"Loaded {len(targets)} targets", flush=True)
    print("Fetching source...", flush=True)
    source_cache = {}
    for t in targets:
        mod = t["module"]
        if mod not in source_cache:
            source_cache[mod] = fetch_source(mod)
    print(f"Fetched {sum(1 for v in source_cache.values() if v)}/{len(source_cache)}\n",
          flush=True)

    print(f"S values: {S_VALUES}", flush=True)
    print(f"Rounds per S: {N_ROUNDS}", flush=True)
    for s in S_VALUES:
        print(f"  S={s}: budget={N_ROUNDS*s} executions", flush=True)

    start = time.time()
    completed = [0]
    total = len(targets)

    def run_target(target):
        source = source_cache.get(target["module"])
        result = {"module": target["module"]}
        for s in S_VALUES:
            result[f"S={s}"] = run_one(target, source, s)
        completed[0] += 1
        finals = {f"S={s}": result[f"S={s}"]["final"] for s in S_VALUES}
        print(f"  [{completed[0]}/{total}] {target['module']}: {finals}", flush=True)
        return result

    all_results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_target, t): t for t in targets}
        for f in as_completed(futures):
            try:
                all_results.append(f.result())
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\n{'='*60}", flush=True)
    print(f"S ablation (matched rounds = {N_ROUNDS}):", flush=True)
    print(f"{'S':<6} {'Budget':<8} {'Mean':>8} {'vs S=1':>8}", flush=True)
    s1_vals = [r["S=1"]["final"] for r in all_results]
    for s in S_VALUES:
        vals = [r[f"S={s}"]["final"] for r in all_results]
        delta = np.mean(vals) - np.mean(s1_vals)
        print(f"  {s:<4} {N_ROUNDS*s:<8} {np.mean(vals):>8.1f} {delta:>+8.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | Time: {elapsed:.0f}s", flush=True)

    from pathlib import Path
    outdir = Path("results/ablations")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "ablation_S_matched.json", "w") as f:
        json.dump({"n_rounds": N_ROUNDS, "results": all_results, "cost": cost}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
