"""Head-to-head strategy comparison: cumulative coverage over N steps.

Strategies:
  1. Random — pick uniformly from K candidates
  2. Greedy — ask LLM which candidate covers the most new branches
  3. Curiosity (logprob) — pick highest token-level logprob entropy
  4. Curiosity (contrastive) — ask LLM to rank by exploration value, pick #1
  5. Oracle — execute all K, pick the one with highest actual gain

Each strategy runs independently on the same functions with the same budget.
Coverage curve tracked at every step.

Usage:
    python run_head2head.py --num-functions 5 --budget 10 --K 3   # smoke test
    python run_head2head.py                                        # full run
"""

import argparse
import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import (
    generate_with_model, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.benchmarks import load_benchmark
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.info_gain import _build_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Functions identified as having corridor structure (validation gates → deep logic)
# Corridor function names (keys depend on load order, so match by name)
CORRIDOR_NAMES = {
    "isRegexPattern",          # code snippet filters → 30+ regex checks
    "artworkAreaR6",           # empty dim checks → char-by-char parser
    "getIntervalRange",        # OID validation → bitmask branching
    "optim_lr_sched_updater",  # None check → two-phase iteration
    "dig_2d",                  # game-over/bomb/revealed checks → recursive flood-fill
    "parse_change",            # empty/prefix checks → nested key/path branching
    "z_defaults",              # channels check → 20+ config defaults
    "get_codec_rank_static",   # empty check → codec cascade + bitrate parsing
    "CheckIfItIsHotspot",      # type coercion → chr/coordinate logic
    "score_v2",                # 5 asserts → contract success/failure nested logic
}

STRATEGIES = ["random", "greedy", "curiosity_sampling", "curiosity_contrastive", "oracle"]


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head strategy comparison")
    parser.add_argument("--num-functions", type=int, default=30)
    parser.add_argument("--budget", type=int, default=15, help="Steps per strategy")
    parser.add_argument("--K", type=int, default=5, help="Candidates per step")
    parser.add_argument("--min-complexity", type=int, default=20)
    parser.add_argument("--benchmark", default="ult")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_random(candidates, func_name, source, test_history, runner):
    """Pick uniformly at random."""
    return random.choice(candidates)


def select_greedy(candidates, func_name, source, test_history, runner):
    """Ask LLM which candidate will cover the most new branches."""
    code_section, history_str = _build_context(func_name, source, test_history)

    if test_history:
        recent = test_history[-5:]
        history_str = "Previous tests and coverage results:\n"
        for tc, res in recent:
            history_str += (f"  {tc} → new_branches={res.new_branches}, "
                           f"output={res.output or res.exception}\n")

    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""Given this function:
{code_section}

{history_str}
Total branches covered: {runner.get_cumulative_coverage()}

Which of these test calls is MOST LIKELY to cover NEW branches not yet covered?

Candidates:
{cand_list}

Respond with ONLY the number (1-{len(candidates)}) of the best candidate."""

    response = generate_with_model(config.MODEL, prompt, temperature=0.3,
                                   max_tokens=20)
    # Parse number
    numbers = re.findall(r'\d+', response)
    if numbers:
        idx = int(numbers[0]) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]  # fallback to first


def select_curiosity_sampling(candidates, func_name, source, test_history, runner):
    """Pick candidate with highest output prediction entropy (original estimator).

    Uses the same model (Gemini) for S predictions at temp=0.9.
    """
    from curiosity_explorer.explorer.info_gain import estimate_output_entropy

    scores = {}
    for cand in candidates:
        scores[cand] = estimate_output_entropy(
            func_name, source, test_history, cand, S=8
        )

    return max(candidates, key=lambda c: scores.get(c, 0))


def select_curiosity_contrastive(candidates, func_name, source, test_history, runner):
    """Ask LLM to rank candidates by exploration value, pick #1."""
    code_section, history_str = _build_context(func_name, source, test_history)

    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""Given this function:
{code_section}

{history_str}

Rank these test calls from MOST to LEAST likely to discover NEW, previously-unseen behavior (new branches, new code paths, new edge cases).

Candidates:
{cand_list}

Respond with ONLY the ranking as comma-separated numbers (best first).
Example: 3,1,5,2,4"""

    response = generate_with_model(config.MODEL, prompt, temperature=0.3,
                                   max_tokens=50)

    # Parse — pick the first valid number
    numbers = re.findall(r'\d+', response)
    for n_str in numbers:
        idx = int(n_str) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def select_oracle(candidates, func_name, source, test_history, runner):
    """Execute all candidates, pick the one with highest actual gain."""
    best_cand = candidates[0]
    best_gain = -1

    for c in candidates:
        temp_runner = CoverageRunner(func_name, source)
        temp_runner.cumulative_branches = set(runner.cumulative_branches)
        result = temp_runner.run_test(c)
        if result.new_branches > best_gain:
            best_gain = result.new_branches
            best_cand = c

    return best_cand


STRATEGY_FNS = {
    "random": select_random,
    "greedy": select_greedy,
    "curiosity_sampling": select_curiosity_sampling,
    "curiosity_contrastive": select_curiosity_contrastive,
    "oracle": select_oracle,
}


# ---------------------------------------------------------------------------
# Run one strategy on one function
# ---------------------------------------------------------------------------

def run_strategy(func_name, source, strategy_name, budget, K, seed):
    """Run a strategy for `budget` steps, return coverage curve."""
    random.seed(seed)
    select_fn = STRATEGY_FNS[strategy_name]

    runner = CoverageRunner(func_name, source)
    test_history = []
    coverage_curve = []

    for step in range(budget):
        # Generate K candidates
        candidates = generate_test_candidates(
            func_name, source, test_history=test_history, K=K
        )
        if not candidates:
            # No candidates — record same coverage and continue
            coverage_curve.append(runner.get_cumulative_coverage())
            continue

        # Select one candidate using the strategy
        selected = select_fn(candidates, func_name, source, test_history, runner)

        # Execute it
        result = runner.run_test(selected)
        test_history.append((selected, result))

        # Record cumulative coverage
        coverage_curve.append(runner.get_cumulative_coverage())

    return coverage_curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("Head-to-Head Strategy Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"  Candidate gen model: {config.MODEL}", flush=True)
    print(f"  Logprob model: {config.LOGPROB_MODEL}", flush=True)
    print(f"  Functions: {args.num_functions} (complexity >= {args.min_complexity})",
          flush=True)
    print(f"  Budget: {args.budget} steps, K={args.K} candidates", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)
    print(f"  Corridor names: {len(CORRIDOR_NAMES)}", flush=True)
    print("=" * 70, flush=True)

    # Connectivity
    print("\nConnectivity...", flush=True)
    for m in [config.MODEL, config.LOGPROB_MODEL]:
        test = generate_with_model(m, "Say 'ok'", temperature=0.3, max_tokens=10)
        print(f"  {m}: {'OK' if test else 'FAILED'}", flush=True)
        if not test:
            return

    # Load functions — guarantee all corridor functions are included
    all_programs = load_benchmark(
        args.benchmark, min_complexity=args.min_complexity,
        max_functions=500,
    )
    # Split into corridor and non-corridor
    corridor_funcs = {k: p for k, p in all_programs.items()
                      if p["func_name"] in CORRIDOR_NAMES}
    non_corridor_all = {k: p for k, p in all_programs.items()
                        if p["func_name"] not in CORRIDOR_NAMES}
    # Sample non-corridor to fill remaining slots
    non_corridor_needed = max(0, args.num_functions - len(corridor_funcs))
    rng = random.Random(args.seed)
    non_corridor_keys = list(non_corridor_all.keys())
    rng.shuffle(non_corridor_keys)
    non_corridor_funcs = {k: non_corridor_all[k]
                          for k in non_corridor_keys[:non_corridor_needed]}
    # Merge
    programs = {}
    programs.update(corridor_funcs)
    programs.update(non_corridor_funcs)
    print(f"  Corridor: {len(corridor_funcs)}, Non-corridor: {len(non_corridor_funcs)}",
          flush=True)

    # Run all strategies on all functions
    all_results = []
    total = len(programs)

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        complexity = prog.get("metadata", {}).get("cyclomatic_complexity", 0)
        is_corridor = func_name in CORRIDOR_NAMES
        tag = "CORRIDOR" if is_corridor else "other"

        print(f"\n[{i+1}/{total}] {key} ({func_name}, c={complexity}, {tag})",
              flush=True)

        func_results = {
            "func_key": key,
            "func_name": func_name,
            "complexity": complexity,
            "is_corridor": is_corridor,
            "strategies": {},
        }

        # Run all strategies in parallel (they are independent)
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, strategy,
                                args.budget, args.K, args.seed): strategy
                for strategy in STRATEGIES
            }
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    curve = future.result()
                except Exception as e:
                    log.warning(f"Strategy {strategy} failed: {e}")
                    curve = []
                func_results["strategies"][strategy] = curve
                final_cov = curve[-1] if curve else 0
                print(f"  {strategy:<25} final_coverage={final_cov}", flush=True)

        all_results.append(func_results)

    # === ANALYSIS ===
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    # Overall: mean final coverage by strategy
    print(f"\n  OVERALL (n={len(all_results)}):", flush=True)
    print(f"  {'Strategy':<25} {'Mean final cov':>15} {'Mean cov@5':>12} {'Mean cov@10':>12}",
          flush=True)
    print(f"  {'─' * 66}", flush=True)
    for s in STRATEGIES:
        finals = [r["strategies"][s][-1] for r in all_results if r["strategies"][s]]
        at5 = [r["strategies"][s][4] for r in all_results
               if r["strategies"][s] and len(r["strategies"][s]) >= 5]
        at10 = [r["strategies"][s][9] for r in all_results
                if r["strategies"][s] and len(r["strategies"][s]) >= 10]
        print(f"  {s:<25} {mean(finals):>15.1f} {mean(at5):>12.1f} {mean(at10):>12.1f}",
              flush=True)

    # Corridor vs non-corridor
    for subset_name, is_corr in [("CORRIDOR", True), ("NON-CORRIDOR", False)]:
        subset = [r for r in all_results if r["is_corridor"] == is_corr]
        if not subset:
            continue
        print(f"\n  {subset_name} (n={len(subset)}):", flush=True)
        print(f"  {'Strategy':<25} {'Mean final cov':>15} {'Mean cov@5':>12} {'Mean cov@10':>12}",
              flush=True)
        print(f"  {'─' * 66}", flush=True)
        for s in STRATEGIES:
            finals = [r["strategies"][s][-1] for r in subset if r["strategies"][s]]
            at5 = [r["strategies"][s][4] for r in subset
                   if r["strategies"][s] and len(r["strategies"][s]) >= 5]
            at10 = [r["strategies"][s][9] for r in subset
                    if r["strategies"][s] and len(r["strategies"][s]) >= 10]
            print(f"  {s:<25} {mean(finals):>15.1f} {mean(at5):>12.1f} {mean(at10):>12.1f}",
                  flush=True)

    # Crossover analysis: at which step does curiosity overtake greedy?
    print(f"\n  CROSSOVER ANALYSIS (curiosity_sampling vs greedy):", flush=True)
    for subset_name, is_corr in [("CORRIDOR", True), ("NON-CORRIDOR", False)]:
        subset = [r for r in all_results if r["is_corridor"] == is_corr]
        if not subset:
            continue

        # Per-step mean coverage
        print(f"\n  {subset_name} — step-by-step mean coverage:", flush=True)
        print(f"  {'Step':>6}", end="", flush=True)
        for s in ["greedy", "curiosity_sampling", "curiosity_contrastive", "random", "oracle"]:
            print(f" {s[:12]:>13}", end="", flush=True)
        print(flush=True)

        budget = args.budget
        for step in range(budget):
            print(f"  {step+1:>6}", end="", flush=True)
            for s in ["greedy", "curiosity_sampling", "curiosity_contrastive", "random", "oracle"]:
                vals = [r["strategies"][s][step] for r in subset
                        if r["strategies"][s] and len(r["strategies"][s]) > step]
                print(f" {mean(vals):>13.1f}", end="", flush=True)
            print(flush=True)

    # Win rate: how often does curiosity beat greedy (final coverage)?
    print(f"\n  WIN RATES (final coverage):", flush=True)
    for cmp_strategy in ["curiosity_sampling", "curiosity_contrastive"]:
        for subset_name, is_corr in [("ALL", None), ("CORRIDOR", True), ("NON-CORRIDOR", False)]:
            if is_corr is None:
                subset = all_results
            else:
                subset = [r for r in all_results if r["is_corridor"] == is_corr]
            if not subset:
                continue

            wins = 0
            losses = 0
            ties = 0
            for r in subset:
                c_curve = r["strategies"][cmp_strategy]
                g_curve = r["strategies"]["greedy"]
                if not c_curve or not g_curve:
                    continue
                if c_curve[-1] > g_curve[-1]:
                    wins += 1
                elif c_curve[-1] < g_curve[-1]:
                    losses += 1
                else:
                    ties += 1
            total_cmp = wins + losses + ties
            print(f"  {cmp_strategy} vs greedy [{subset_name}]: "
                  f"W={wins} L={losses} T={ties} "
                  f"(win rate={wins/total_cmp:.0%})" if total_cmp > 0 else "",
                  flush=True)

    # Cost
    cost = get_cost()
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}", flush=True)
    print(f"  Cost: ${cost['total_cost_usd']:.4f} | "
          f"Calls: {cost['api_calls']} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if cost.get("per_model"):
        for model, usage in cost["per_model"].items():
            print(f"    {model}: {usage['api_calls']} calls, ${usage['cost_usd']:.4f}",
                  flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "head2head_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "candidate_model": config.MODEL,
                "logprob_model": config.LOGPROB_MODEL,
                "benchmark": args.benchmark,
                "num_functions": args.num_functions,
                "budget": args.budget,
                "K": args.K,
                "min_complexity": args.min_complexity,
                "seed": args.seed,
                "strategies": STRATEGIES,
                "corridor_names": list(CORRIDOR_NAMES),
            },
            "results": all_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


start_time = time.time()

if __name__ == "__main__":
    main()
