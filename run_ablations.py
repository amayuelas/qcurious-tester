"""Run ablation studies on RepoExploreBench with Gemini Flash.

Ablations:
  1. gamma: 0, 0.5, 1.0
  2. K (plans): 1, 3, 5
  3. S (scripts per plan): 1, 3, 5
  4. exec_budget: 8, 16, 24, 32

Each ablation varies one parameter, holds others at default.
Only runs cov_qvalue (our method) + random (baseline).

Usage:
    python run_ablations.py --ablation gamma
    python run_ablations.py --ablation K
    python run_ablations.py --ablation S
    python run_ablations.py --ablation budget
    python run_ablations.py --ablation all
"""

import argparse
import random as _random
import re
import time
import json
import logging
import statistics
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
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Ablation studies")
    p.add_argument("--ablation", required=True,
                   choices=["gamma", "K", "S", "budget", "all"])
    p.add_argument("--parallel", type=int, default=4)
    return p.parse_args()


def fetch_source(module_name):
    import subprocess
    cmd = (f"docker run --rm {DOCKER_IMAGE} python3 -c "
           f"\"import inspect, {module_name}; print(inspect.getsource({module_name}))\"")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def gen_standard(module, source, hist, K):
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    ctx = f"Module: {module}\n{code_ctx}"
    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"
    prompt = (f"{ctx}\n{h}\nWrite a test script (5-15 lines). "
              f"Import from the module and print results.\n"
              f"Respond with ONLY executable Python code.\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def run_one(target, source, strategy, seed, exec_budget, K, plan_length, gamma):
    """Run one strategy with given parameters."""
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

    while executions < exec_budget:
        if strategy == "cov_qvalue":
            scripts = generate_coverage_qvalue(source, module, hist, cov_map,
                                                K=K, plan_length=plan_length,
                                                gamma=gamma)
        else:
            scripts = gen_standard(module, source, hist, 3)

        if not scripts:
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            continue

        if strategy == "cov_qvalue":
            for plan_script in scripts:
                if executions >= exec_budget:
                    break
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
        else:
            selected = _random.choice(scripts)
            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())

    stats = runner.get_stats()
    return {
        "final": stats["branches"],
        "branch_curve": branch_curve,
    }


def run_ablation(name, param_name, param_values, defaults, targets, source_cache, parallel):
    """Run one ablation study."""
    print(f"\n{'='*70}", flush=True)
    print(f"ABLATION: {name} ({param_name} = {param_values})", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = []
    completed = [0]
    total_jobs = len(targets) * len(param_values)

    def run_target(target, param_val):
        source = source_cache.get(target["module"])
        params = dict(defaults)
        params[param_name] = param_val

        result_random = run_one(target, source, "random", 42,
                                 params["exec_budget"], 3, 3, 0.5)
        result_qvalue = run_one(target, source, "cov_qvalue", 42,
                                 params["exec_budget"], params["K"],
                                 params["plan_length"], params["gamma"])
        completed[0] += 1
        print(f"  [{completed[0]}/{total_jobs}] {target['module']} "
              f"{param_name}={param_val}: random={result_random['final']}, "
              f"cov_qvalue={result_qvalue['final']}", flush=True)
        return {
            "module": target["module"],
            param_name: param_val,
            "random": result_random,
            "cov_qvalue": result_qvalue,
        }

    jobs = [(t, v) for v in param_values for t in targets]

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(run_target, t, v): (t, v) for t, v in jobs}
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as e:
                t, v = futures[future]
                print(f"  ERROR {t['module']} {param_name}={v}: {e}", flush=True)

    # Analyze
    print(f"\n--- {name} Results ---", flush=True)
    print(f"{'Value':<10} {'Random':>10} {'CovQValue':>10} {'Delta':>10} {'Wins':>8}",
          flush=True)
    for v in param_values:
        v_results = [r for r in all_results if r[param_name] == v]
        rand_vals = [r["random"]["final"] for r in v_results]
        qv_vals = [r["cov_qvalue"]["final"] for r in v_results]
        deltas = [q - r for q, r in zip(qv_vals, rand_vals)]
        wins = sum(1 for d in deltas if d > 0)
        print(f"{v:<10} {np.mean(rand_vals):>10.1f} {np.mean(qv_vals):>10.1f} "
              f"{np.mean(deltas):>+10.1f} {wins}/{len(deltas):>5}", flush=True)

    return all_results


def main():
    args = parse_args()
    reset_cost()

    targets = load_benchmark()
    print(f"Loaded {len(targets)} targets", flush=True)

    # Fetch source
    print("Fetching source code...", flush=True)
    source_cache = {}
    for t in targets:
        mod = t["module"]
        if mod not in source_cache:
            source_cache[mod] = fetch_source(mod)
    print(f"Fetched {sum(1 for v in source_cache.values() if v)}/{len(source_cache)}",
          flush=True)

    defaults = {"gamma": 0.5, "K": 3, "plan_length": 3, "exec_budget": 24}

    ablations_to_run = []
    if args.ablation in ("gamma", "all"):
        ablations_to_run.append(("gamma", "gamma", [0.0, 0.5, 1.0]))
    if args.ablation in ("K", "all"):
        ablations_to_run.append(("K_plans", "K", [1, 3, 5]))
    if args.ablation in ("S", "all"):
        ablations_to_run.append(("S_plan_length", "plan_length", [1, 3, 5]))
    if args.ablation in ("budget", "all"):
        ablations_to_run.append(("exec_budget", "exec_budget", [8, 16, 24, 32]))

    start = time.time()
    all_ablation_results = {}

    for name, param_name, param_values in ablations_to_run:
        results = run_ablation(name, param_name, param_values, defaults,
                                targets, source_cache, args.parallel)
        all_ablation_results[name] = results

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\nTotal cost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    from pathlib import Path
    outdir = Path("results/ablations")
    outdir.mkdir(parents=True, exist_ok=True)

    for name, results in all_ablation_results.items():
        with open(outdir / f"ablation_{name}.json", "w") as f:
            json.dump({"ablation": name, "results": results,
                        "cost": cost, "elapsed": round(elapsed, 1)},
                      f, indent=2)
        print(f"Saved ablation_{name}.json", flush=True)


if __name__ == "__main__":
    main()
