"""Ablation: diversity hints vs Q-value selection.

Compares:
  cov_diverse: K=3 diverse hints, random selection (no Q-value)
  cov_qvalue: K=3 diverse hints + Q-value selection
  cov_nodiversity: K=3 same prompt, Q-value selection
  cov_greedy: K=3 same prompt, random selection (existing baseline)
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
    CoverageMap, generate_coverage_greedy, generate_coverage_qvalue,
    _generate_k_plans, _score_and_select_plan, _parse_script, _parse_plan,
)
from curiosity_explorer.benchmarks.repo_explore_bench import (
    load_benchmark, DOCKER_IMAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

K = 3
PLAN_LENGTH = 3
GAMMA = 0.5
EXEC_BUDGET = 24
STRATEGIES = ["cov_greedy", "cov_diverse", "cov_qvalue", "cov_nodiversity"]


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


def gen_plans_no_diversity(source, module_name, hist, cov_map, K=3, plan_length=3):
    """Generate K plans with SAME prompt (no diversity hints)."""
    code_section = f"```python\n{source[:2500]}\n```"
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach
UNCOVERED code paths.

Think about what setup is needed:
Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-10 lines each).
Import from the module and print results.

Format your response as:
### TEST 1
```python
[code]
```
### TEST 2
```python
[code]
```
### TEST 3
```python
[code]
```
"""
    # Same prompt K times — no diversity hints
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=1500)
    plans = [_parse_plan(r) for r in responses if _parse_plan(r)]
    return plans


def run_one(target, source, strategy, seed):
    _random.seed(seed)
    module = target["module"]

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

    while executions < EXEC_BUDGET:
        if strategy == "cov_greedy":
            # Same prompt × K, random selection
            scripts = generate_coverage_greedy(source, module, hist, cov_map, K=K)
            if not scripts:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                continue
            selected = _random.choice(scripts)
            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())

        elif strategy == "cov_diverse":
            # Diverse hints, random selection (no Q-value)
            plans = _generate_k_plans(source, module, hist, cov_map,
                                       K=K, plan_length=PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                continue
            selected_plan = _random.choice(plans)
            for script in selected_plan:
                if executions >= EXEC_BUDGET:
                    break
                result = runner.run_test(script)
                hist.append((script, result))
                cov_map.update(script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())

        elif strategy == "cov_qvalue":
            # Diverse hints + Q-value selection
            scripts = generate_coverage_qvalue(source, module, hist, cov_map,
                                                K=K, plan_length=PLAN_LENGTH,
                                                gamma=GAMMA)
            if not scripts:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                continue
            for script in scripts:
                if executions >= EXEC_BUDGET:
                    break
                result = runner.run_test(script)
                hist.append((script, result))
                cov_map.update(script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())

        elif strategy == "cov_nodiversity":
            # Same prompt × K + Q-value selection
            plans = gen_plans_no_diversity(source, module, hist, cov_map,
                                           K=K, plan_length=PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                continue
            if len(plans) == 1:
                selected_plan = plans[0]
            else:
                selected_plan = _score_and_select_plan(
                    plans, source, module, cov_map, gamma=GAMMA)
            for script in selected_plan:
                if executions >= EXEC_BUDGET:
                    break
                result = runner.run_test(script)
                hist.append((script, result))
                cov_map.update(script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())

    return {
        "final": runner.get_cumulative_coverage(),
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
    print(f"Fetched {sum(1 for v in source_cache.values() if v)}/{len(source_cache)}", flush=True)

    print(f"\nStrategies: {STRATEGIES}", flush=True)
    print(f"Budget: {EXEC_BUDGET}, K={K}, S={PLAN_LENGTH}\n", flush=True)

    start = time.time()
    completed = [0]
    total = len(targets)

    def run_target(target):
        source = source_cache.get(target["module"])
        result = {"module": target["module"]}
        for s in STRATEGIES:
            result[s] = run_one(target, source, s, 42)
        completed[0] += 1
        finals = {s: result[s]["final"] for s in STRATEGIES}
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
    print(f"{'Strategy':<20} {'Mean':>8} {'Wins vs CovGreedy':>18}", flush=True)
    for s in STRATEGIES:
        vals = [r[s]["final"] for r in all_results]
        greedy_vals = [r["cov_greedy"]["final"] for r in all_results]
        wins = sum(1 for v, g in zip(vals, greedy_vals) if v > g)
        print(f"  {s:<18} {np.mean(vals):>8.1f} {wins}/{len(vals):>15}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | Time: {elapsed:.0f}s", flush=True)

    from pathlib import Path
    outdir = Path("results/ablations")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "ablation_diversity.json", "w") as f:
        json.dump({"results": all_results, "cost": cost}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
