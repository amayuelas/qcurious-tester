"""RepoExploreBench runner.

Evaluates exploration strategies on 100 real-world Python modules
across 9 repos, all running in the curiositybench Docker image.

Usage:
    # Full benchmark (100 files × 1 seed × 3 strategies)
    python run_repo_explore_bench.py

    # Quick smoke test
    python run_repo_explore_bench.py --max-targets 3 --seeds 42 --exec-budget 6

    # Single repo
    python run_repo_explore_bench.py --repos click

    # Only key comparison
    python run_repo_explore_bench.py --strategies random cov_qvalue

    # Multiple seeds for tighter CIs
    python run_repo_explore_bench.py --seeds 42 123 456
"""

import argparse
import random as _random
import re
import subprocess
import time
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, _parse_script, _parse_plan,
    generate_plans_for_exec_selection,
)
from curiosity_explorer.benchmarks.repo_explore_bench import (
    load_benchmark, get_benchmark_info, DEFAULT_SEEDS, DOCKER_IMAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

ALL_STRATEGIES = ["random", "greedy", "cov_greedy", "cov_qvalue"]
EXEC_BUDGET = 24
K = 3
PLAN_LENGTH = 3
GAMMA = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="RepoExploreBench runner")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--repos", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--exec-budget", type=int, default=EXEC_BUDGET)
    p.add_argument("--K", type=int, default=K)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--parallel", type=int, default=4,
                   help="Number of targets to run in parallel")
    p.add_argument("--output", default="repo_explore_bench_results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source code fetching
# ---------------------------------------------------------------------------

def fetch_source(module_name):
    """Fetch module source code from Docker image."""
    cmd = (f"docker run --rm {DOCKER_IMAGE} python3 -c "
           f"\"import inspect, {module_name}; print(inspect.getsource({module_name}))\"")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        log.warning(f"Failed to fetch source for {module_name}: {e}")
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def gen_standard(module, source, hist, K):
    """Standard generation — show module info and recent history."""
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


def select_greedy(scripts, module, source, hist):
    """LLM picks the script most likely to cover new code."""
    code_ctx = f"```python\n{source[:2000]}\n```" if source else ""
    ctx = f"Module: {module}\n{code_ctx}"
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    prompt = (f"{ctx}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def gen_cov_greedy(module, source, hist, cov_map, K):
    """Coverage-aware generation — show source + coverage map, target gaps."""
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Previous tests:
{history_str}

Write a test script (5-15 lines) that covers branches NOT YET discovered.
Look at the code and target specific uncovered code paths.
Import from the module and print results.
Respond with ONLY executable Python code.

```python
"""
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_cov_qvalue(module, source, hist, cov_map, K, plan_length, gamma):
    """Generate K plans, score by Q-value, return the best plan."""
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    diversity_hints = [
        "Focus on the MAIN functionality — constructors, primary methods.",
        "Focus on ERROR HANDLING — invalid inputs, edge cases, exceptions.",
        "Focus on INTERACTIONS — create objects, pass them to each other, chain calls.",
        "Focus on CONFIGURATION — different parameter combinations, options, flags.",
        "Focus on RARELY-USED features — optional arguments, deprecated paths.",
    ]

    prompts = []
    for i in range(K):
        hint = diversity_hints[i % len(diversity_hints)]
        prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach
UNCOVERED code paths. {hint}

Think about what setup is needed:
Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-15 lines each).
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
        prompts.append(prompt)

    # Generate K plans
    responses = batch_generate(prompts, temperature=0.9, max_tokens=1500)
    plans = [_parse_plan(r) for r in responses if _parse_plan(r)]

    if not plans:
        return gen_cov_greedy(module, source, hist, cov_map, K)
    if len(plans) == 1:
        return plans[0]

    # Score each plan by Q-value
    return _score_and_select(plans, module, source, cov_map, gamma)


def _score_and_select(plans, module, source, cov_map, gamma):
    """Score plans by Q-value and return the best."""
    code_ctx = f"```python\n{source[:2000]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary()

    def score_plan(idx):
        plan = plans[idx]
        plan_str = ""
        for i, s in enumerate(plan):
            plan_str += f"\nStep {i+1}:\n```python\n{s[:200]}\n```\n"

        prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Consider this TEST PLAN (a sequence of {len(plan)} scripts):
{plan_str}

Evaluate by answering TWO questions with just numbers:
1. IMMEDIATE GAIN: Total NEW branches this plan discovers? (0-50)
2. FUTURE VALUE: Additional branches reachable AFTER this plan? (0-50)

Format: immediate, future
Example: 15, 25"""

        resp = generate_with_model("gemini-3-flash-preview", prompt, 0.3, 50)
        nums = re.findall(r'\d+', resp)
        imm = int(nums[0]) if len(nums) >= 1 else 0
        fut = int(nums[1]) if len(nums) >= 2 else 0
        q = imm + gamma * fut
        log.info(f"Plan {idx}: ḡ={imm}, γE[v]={gamma*fut:.1f}, Q={q:.1f}")
        return q

    scores = {}
    with ThreadPoolExecutor(max_workers=len(plans)) as ex:
        futures = {ex.submit(score_plan, i): i for i, _ in enumerate(plans)}
        for f in futures:
            idx = futures[f]
            try:
                scores[idx] = f.result()
            except Exception:
                scores[idx] = 0

    best = max(scores, key=scores.get)
    log.info(f"Selected plan {best} with Q={scores[best]:.1f}")
    return plans[best]


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(target, strategy, seed, exec_budget, K, gamma, source):
    """Run one strategy on one target. Returns {final, curve}."""
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
    line_curve = []

    while executions < exec_budget:
        # --- Generation ---
        if strategy == "cov_greedy":
            scripts = gen_cov_greedy(module, source, hist, cov_map, K)
        elif strategy == "cov_qvalue":
            scripts = gen_cov_qvalue(module, source, hist, cov_map,
                                      K, PLAN_LENGTH, gamma)
        elif strategy == "cov_qvalue_exec":
            scripts = None  # handled below
        else:
            scripts = gen_standard(module, source, hist, K)

        # --- Execution-based Q-value selection ---
        if strategy == "cov_qvalue_exec":
            plans = generate_plans_for_exec_selection(
                source, module, hist, cov_map, K=K, plan_length=PLAN_LENGTH)

            if not plans or executions >= exec_budget:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            # Execute step 1 of each plan — observe actual coverage
            step1_results = []
            for plan in plans:
                if executions >= exec_budget:
                    break
                result = runner.run_test(plan[0])
                hist.append((plan[0], result))
                cov_map.update(plan[0], set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                step1_results.append((plan, result.new_branches))

            # Select the plan whose step 1 discovered the most branches
            if step1_results:
                best_plan, _ = max(step1_results, key=lambda x: x[1])

                # Execute remaining steps of the winning plan
                for plan_script in best_plan[1:]:
                    if executions >= exec_budget:
                        break
                    result = runner.run_test(plan_script)
                    hist.append((plan_script, result))
                    cov_map.update(plan_script, set(), result.new_branches)
                    executions += 1
                    branch_curve.append(runner.get_cumulative_coverage())
                    line_curve.append(runner.get_cumulative_lines())

            continue

        if not scripts:
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())
            continue

        # --- Standard execution ---
        if strategy == "cov_qvalue":
            for plan_script in scripts:
                if executions >= exec_budget:
                    break
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
        else:
            if strategy == "random":
                selected = _random.choice(scripts)
            elif strategy == "greedy":
                selected = select_greedy(scripts, module, source, hist)
            elif strategy == "cov_greedy":
                selected = _random.choice(scripts)
            else:
                selected = scripts[0]

            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())

    stats = runner.get_stats()

    # Serialize execution trace
    trace = []
    for script, result in hist:
        trace.append({
            "script": script,
            "output": result.output,
            "exception": result.exception,
            "new_branches": result.new_branches,
            "new_lines": result.new_lines,
            "passed": result.passed,
        })

    return {
        "final": stats["branches"],
        "final_lines": stats["lines"],
        "pass_rate": stats["pass_rate"],
        "pass_count": stats["pass_count"],
        "fail_count": stats["fail_count"],
        "branch_curve": branch_curve,
        "line_curve": line_curve,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyze_results(all_results, strategies):
    """Compute summary statistics and paired tests."""
    from scipy import stats as sp_stats

    analysis = {}

    for s in strategies:
        vals = [r["strategies"][s]["final"] for r in all_results
                if s in r["strategies"]]
        analysis[s] = {
            "mean": statistics.mean(vals) if vals else 0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0,
            "se": statistics.stdev(vals) / len(vals)**0.5 if len(vals) > 1 else 0,
            "n": len(vals),
        }

    # Paired comparisons vs random
    if "random" in strategies:
        analysis["paired_vs_random"] = {}
        for s in strategies:
            if s == "random":
                continue
            deltas = []
            for r in all_results:
                if "random" in r["strategies"] and s in r["strategies"]:
                    d = r["strategies"][s]["final"] - r["strategies"]["random"]["final"]
                    deltas.append(d)
            if len(deltas) < 2:
                continue
            md = statistics.mean(deltas)
            sd = statistics.stdev(deltas)
            se = sd / len(deltas)**0.5
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            ties = sum(1 for d in deltas if d == 0)

            if sd > 0:
                t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
                cohens_d = md / sd
            else:
                t_stat, p_val, cohens_d = 0, 1.0, 0

            analysis["paired_vs_random"][s] = {
                "mean_delta": md, "se": se,
                "wins": wins, "losses": losses, "ties": ties,
                "t_stat": t_stat, "p_value": p_val,
                "cohens_d": cohens_d, "n": len(deltas),
            }

    # Pairwise: cov_qvalue vs cov_greedy
    if "cov_qvalue" in strategies and "cov_greedy" in strategies:
        deltas = []
        for r in all_results:
            if "cov_qvalue" in r["strategies"] and "cov_greedy" in r["strategies"]:
                d = (r["strategies"]["cov_qvalue"]["final"] -
                     r["strategies"]["cov_greedy"]["final"])
                deltas.append(d)
        if len(deltas) >= 2 and statistics.stdev(deltas) > 0:
            t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
            analysis["qvalue_vs_greedy"] = {
                "mean_delta": statistics.mean(deltas),
                "se": statistics.stdev(deltas) / len(deltas)**0.5,
                "t_stat": t_stat, "p_value": p_val,
                "n": len(deltas),
            }

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    targets = load_benchmark(repos=args.repos, max_targets=args.max_targets)
    bench_info = get_benchmark_info()
    strategies = args.strategies
    seeds = args.seeds

    total_runs = len(targets) * len(strategies) * len(seeds)
    print("=" * 70, flush=True)
    print(f"RepoExploreBench v{bench_info['version']}", flush=True)
    print(f"  Targets: {len(targets)} files across "
          f"{len(set(t['repo'] for t in targets))} repos", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Exec budget: {args.exec_budget} per run", flush=True)
    print(f"  Total runs: {total_runs}", flush=True)
    print("=" * 70, flush=True)

    # Connectivity check
    t = generate_with_model(config.MODEL, "Say ok", 0.3, 100)
    print(f"  Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    # Pre-fetch source code for all targets
    print(f"\n  Fetching source code...", flush=True)
    source_cache = {}
    for target in targets:
        mod = target["module"]
        if mod not in source_cache:
            source_cache[mod] = fetch_source(mod)
            status = f"{len(source_cache[mod])} chars" if source_cache[mod] else "FAILED"
            print(f"    {mod}: {status}", flush=True)
    print(f"  Fetched {sum(1 for v in source_cache.values() if v)}/{len(source_cache)} modules",
          flush=True)

    start = time.time()
    completed = [0]  # mutable counter for thread safety

    def run_one_target(i, target, seed):
        """Run all strategies on one target. Returns result dict."""
        source = source_cache.get(target["module"])
        run_result = {
            "module": target["module"],
            "repo": target["repo"],
            "seed": seed,
            "strategies": {},
        }
        for strategy in strategies:
            result = run_strategy(target, strategy, seed,
                                   args.exec_budget, args.K, args.gamma,
                                   source)
            run_result["strategies"][strategy] = result
        completed[0] += 1
        finals = {s: run_result["strategies"][s]["final"] for s in strategies}
        print(f"  [{completed[0]}/{len(targets)*len(seeds)}] "
              f"{target['module']} seed={seed}: {finals}", flush=True)
        return run_result

    # Build list of (index, target, seed) jobs
    jobs = [(i, target, seed)
            for i, target in enumerate(targets)
            for seed in seeds]

    print(f"\nRunning {len(jobs)} targets with {args.parallel} workers...",
          flush=True)

    all_results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_one_target, i, t, s): (i, t, s)
                   for i, t, s in jobs}
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as e:
                i, t, s = futures[future]
                print(f"  ERROR on {t['module']} seed={s}: {e}", flush=True)

    elapsed = time.time() - start
    cost = get_cost()

    # --- Summary table ---
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Module':<35} {'seed':>4}", end="", flush=True)
    for s in strategies:
        print(f" {s[:11]:>12}", end="")
    print(flush=True)
    print("-" * (40 + 13 * len(strategies)), flush=True)

    for r in all_results:
        print(f"{r['module'][:34]:<35} {r['seed']:>4}", end="")
        for s in strategies:
            v = r["strategies"].get(s, {}).get("final", 0)
            print(f" {v:>12}", end="")
        print(flush=True)

    # --- Statistics ---
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICS", flush=True)
    print(f"{'=' * 70}", flush=True)

    analysis = analyze_results(all_results, strategies)

    print(f"\nPer-strategy means (n={analysis[strategies[0]]['n']}):", flush=True)
    for s in strategies:
        a = analysis[s]
        # Line coverage and pass rate
        lines = [r["strategies"][s].get("final_lines", 0) for r in all_results
                 if s in r["strategies"]]
        pass_rates = [r["strategies"][s].get("pass_rate", 0) for r in all_results
                      if s in r["strategies"]]
        mean_lines = statistics.mean(lines) if lines else 0
        mean_pr = statistics.mean(pass_rates) if pass_rates else 0
        print(f"  {s:<20} branches={a['mean']:>6.1f} ± {a['se']:.1f}  "
              f"lines={mean_lines:.1f}  pass_rate={mean_pr:.0%}", flush=True)

    if "paired_vs_random" in analysis:
        print(f"\nPaired vs random:", flush=True)
        for s, a in analysis["paired_vs_random"].items():
            sig = "***" if a["p_value"] < 0.001 else ("**" if a["p_value"] < 0.01
                    else ("*" if a["p_value"] < 0.05 else ""))
            print(f"  {s:<20} Δ={a['mean_delta']:>+6.1f} ± {a['se']:.1f}  "
                  f"W={a['wins']} L={a['losses']} T={a['ties']}  "
                  f"p={a['p_value']:.4f} d={a['cohens_d']:.2f} {sig}", flush=True)

    if "qvalue_vs_greedy" in analysis:
        a = analysis["qvalue_vs_greedy"]
        sig = "*" if a["p_value"] < 0.05 else ""
        print(f"\ncov_qvalue vs cov_greedy: "
              f"Δ={a['mean_delta']:>+.1f} ± {a['se']:.1f}  "
              f"p={a['p_value']:.4f} {sig}", flush=True)

    # --- Per-repo breakdown ---
    print(f"\nPer-repo means:", flush=True)
    repos = sorted(set(r["repo"] for r in all_results))
    for repo in repos:
        repo_results = [r for r in all_results if r["repo"] == repo]
        print(f"\n  {repo} ({len(repo_results)} runs):", flush=True)
        for s in strategies:
            vals = [r["strategies"][s]["final"] for r in repo_results
                    if s in r["strategies"]]
            if vals:
                print(f"    {s:<20} mean={statistics.mean(vals):.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # --- Save ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = config.RESULTS_DIR / args.output
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": bench_info,
            "config": {
                "strategies": strategies, "seeds": seeds,
                "exec_budget": args.exec_budget, "K": args.K,
                "gamma": args.gamma,
            },
            "results": all_results,
            "analysis": {k: v for k, v in analysis.items()
                          if k != "paired_vs_random"},
            "paired_vs_random": analysis.get("paired_vs_random", {}),
            "cost": cost,
            "elapsed": round(elapsed, 1),
        }, f, indent=2, default=str)
    print(f"Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
