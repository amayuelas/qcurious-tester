"""Head-to-head v2: Improved generation + selection strategies.

New strategies:
  - diverse_random: Diverse generation + random selection
  - diverse_art: Diverse generation + ART (most-distant) selection
  - diverse_curiosity: Diverse generation + sampling entropy selection
  - diverse_art_curiosity: Diverse generation + ART+entropy combined

Compared against original strategies:
  - random: Standard generation + random selection
  - greedy: Standard generation + greedy selection
  - curiosity_sampling: Standard generation + entropy selection
  - oracle: Standard generation + oracle selection

Usage:
    python run_head2head_v2.py --num-functions 5 --budget 10 --K 5  # smoke
    python run_head2head_v2.py                                       # full
    python run_head2head_v2.py --toy                                 # toy programs
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
    generate_with_model, batch_generate, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.benchmarks import load_benchmark
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.diverse_gen import generate_diverse_candidates
from curiosity_explorer.explorer.art_selection import (
    select_most_distant, select_art_with_entropy,
)
from curiosity_explorer.explorer.info_gain import (
    _build_context, estimate_output_entropy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Corridor function names
CORRIDOR_NAMES = {
    "isRegexPattern", "artworkAreaR6", "getIntervalRange",
    "optim_lr_sched_updater", "dig_2d", "parse_change", "z_defaults",
    "get_codec_rank_static", "CheckIfItIsHotspot", "score_v2",
}

# Functions where oracle >> greedy (selection actually matters)
HIGH_GAP_NAMES = {
    "isCardMatch", "dig_2d", "get_codec_rank_static", "isRegexPattern",
    "CheckIfItIsHotspot", "getIntervalRange", "score_v2", "statusInterpreter",
}

STRATEGIES = [
    # Original strategies (standard generation)
    "random", "greedy", "curiosity_sampling", "oracle",
    # New strategies (diverse generation)
    "diverse_random", "diverse_art", "diverse_curiosity",
    "diverse_art_curiosity",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head v2: improved strategies")
    parser.add_argument("--num-functions", type=int, default=30)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--min-complexity", type=int, default=20)
    parser.add_argument("--benchmark", default="ult")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy", action="store_true", help="Run on toy programs instead")
    parser.add_argument("--targeted", action="store_true",
                        help="Run only on high-gap functions where selection matters")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _select_greedy(candidates, func_name, source, test_history, runner):
    code_section, history_str = _build_context(func_name, source, test_history)
    if test_history:
        recent = test_history[-5:]
        history_str = "Previous tests and coverage results:\n"
        for tc, res in recent:
            history_str += (f"  {tc} → new_branches={res.new_branches}, "
                           f"output={res.output or res.exception}\n")
    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (f"Given this function:\n{code_section}\n\n{history_str}\n"
              f"Total branches covered: {runner.get_cumulative_coverage()}\n\n"
              f"Which test is MOST LIKELY to cover NEW branches? "
              f"Respond with ONLY the number.\n\nCandidates:\n{cand_list}")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _select_oracle(candidates, func_name, source, runner):
    best, best_gain = candidates[0], -1
    for c in candidates:
        tr = CoverageRunner(func_name, source)
        tr.cumulative_branches = set(runner.cumulative_branches)
        r = tr.run_test(c)
        if r.new_branches > best_gain:
            best, best_gain = c, r.new_branches
    return best


def _score_entropy(candidates, func_name, source, test_history, S=8):
    """Score all candidates with sampling entropy."""
    scores = {}
    for c in candidates:
        scores[c] = estimate_output_entropy(func_name, source, test_history, c, S=S)
    return scores


# ---------------------------------------------------------------------------
# Run one strategy
# ---------------------------------------------------------------------------

def run_strategy(func_name, source, strategy, budget, K, S, seed):
    """Run a single strategy, return coverage curve."""
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []

    use_diverse = strategy.startswith("diverse_")

    for step in range(budget):
        # Generate candidates
        if use_diverse:
            candidates = generate_diverse_candidates(
                func_name, source, test_history=test_history, K=K
            )
        else:
            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K
            )

        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            continue

        # Select based on strategy
        if strategy == "random":
            selected = random.choice(candidates)

        elif strategy == "greedy":
            selected = _select_greedy(candidates, func_name, source,
                                      test_history, runner)

        elif strategy == "curiosity_sampling":
            scores = _score_entropy(candidates, func_name, source,
                                    test_history, S)
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy == "oracle":
            selected = _select_oracle(candidates, func_name, source, runner)

        elif strategy == "diverse_random":
            selected = random.choice(candidates)

        elif strategy == "diverse_art":
            selected = select_most_distant(candidates, test_history)

        elif strategy == "diverse_curiosity":
            scores = _score_entropy(candidates, func_name, source,
                                    test_history, S)
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy == "diverse_art_curiosity":
            scores = _score_entropy(candidates, func_name, source,
                                    test_history, S)
            selected = select_art_with_entropy(candidates, test_history,
                                               scores, alpha=0.5)
        else:
            selected = candidates[0]

        # Execute
        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("Head-to-Head v2: Improved Generation + Selection", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)

    # Load functions
    if args.toy:
        from curiosity_explorer.benchmarks.toy_programs import TOY_PROGRAMS
        programs = {}
        for key, prog in TOY_PROGRAMS.items():
            programs[key] = {
                "func_name": prog["func_name"],
                "source": prog["source"],
                "metadata": {"cyclomatic_complexity": prog.get("expected_branches", 0)},
            }
        print(f"  Toy programs: {len(programs)}", flush=True)
    else:
        all_programs = load_benchmark(
            args.benchmark, min_complexity=args.min_complexity, max_functions=500,
        )
        if args.targeted:
            # Only functions where selection matters (oracle >> greedy)
            programs = {k: p for k, p in all_programs.items()
                        if p["func_name"] in HIGH_GAP_NAMES}
            print(f"  Targeted: {len(programs)} high-gap functions", flush=True)
        else:
            corridor_funcs = {k: p for k, p in all_programs.items()
                              if p["func_name"] in CORRIDOR_NAMES}
            non_corridor_all = {k: p for k, p in all_programs.items()
                                if p["func_name"] not in CORRIDOR_NAMES}
            non_corridor_needed = max(0, args.num_functions - len(corridor_funcs))
            rng = random.Random(args.seed)
            nc_keys = list(non_corridor_all.keys())
            rng.shuffle(nc_keys)
            non_corridor = {k: non_corridor_all[k]
                            for k in nc_keys[:non_corridor_needed]}
            programs = {}
            programs.update(corridor_funcs)
            programs.update(non_corridor)
        n_corr = sum(1 for p in programs.values()
                     if p["func_name"] in CORRIDOR_NAMES)
        n_non = len(programs) - n_corr
        print(f"  Functions: {len(programs)} (corridor={n_corr}, other={n_non})",
              flush=True)

    print("=" * 70, flush=True)

    # Connectivity
    test = generate_with_model(config.MODEL, "Say 'ok'", 0.3, 10)
    print(f"  Model connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []
    total = len(programs)

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        complexity = prog.get("metadata", {}).get("cyclomatic_complexity", 0)
        is_corridor = func_name in CORRIDOR_NAMES

        tag = "CORR" if is_corridor else "other"
        if args.toy:
            tag = f"depth={complexity}"

        print(f"\n[{i+1}/{total}] {key} ({func_name}, {tag})", flush=True)

        func_results = {
            "func_key": key,
            "func_name": func_name,
            "complexity": complexity,
            "is_corridor": is_corridor,
            "strategies": {},
        }

        # Run all strategies in parallel
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, s,
                                args.budget, args.K, args.S, args.seed): s
                for s in STRATEGIES
            }
            for future in as_completed(futures):
                s = futures[future]
                try:
                    curve = future.result()
                except Exception as e:
                    log.warning(f"{s} failed: {e}")
                    curve = []
                func_results["strategies"][s] = curve
                final = curve[-1] if curve else 0
                print(f"  {s:<25} final={final}", flush=True)

        all_results.append(func_results)

    elapsed = time.time() - start
    cost = get_cost()

    # === ANALYSIS ===
    def mean(v):
        return sum(v) / len(v) if v else 0

    # Group strategies
    orig = ["random", "greedy", "curiosity_sampling", "oracle"]
    new = ["diverse_random", "diverse_art", "diverse_curiosity",
           "diverse_art_curiosity"]

    for subset_name, pred in [("OVERALL", lambda r: True),
                               ("CORRIDOR", lambda r: r["is_corridor"]),
                               ("NON-CORRIDOR", lambda r: not r["is_corridor"])]:
        subset = [r for r in all_results if pred(r)]
        if not subset:
            continue
        print(f"\n{'=' * 70}", flush=True)
        print(f"{subset_name} (n={len(subset)})", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  {'Strategy':<25} {'Final':>7} {'@5':>7} {'@10':>7}", flush=True)
        print(f"  {'─' * 50}", flush=True)

        for s in orig + ["---"] + new:
            if s == "---":
                print(f"  {'─' * 50}", flush=True)
                continue
            finals = [r["strategies"][s][-1] for r in subset if r["strategies"][s]]
            at5 = [r["strategies"][s][4] for r in subset
                   if r["strategies"][s] and len(r["strategies"][s]) >= 5]
            at10 = [r["strategies"][s][9] for r in subset
                    if r["strategies"][s] and len(r["strategies"][s]) >= 10]
            print(f"  {s:<25} {mean(finals):>7.1f} {mean(at5):>7.1f} "
                  f"{mean(at10):>7.1f}", flush=True)

    # Win rates: new strategies vs original greedy
    print(f"\n{'=' * 70}", flush=True)
    print("WIN RATES vs GREEDY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cmp in new:
        for subset_name, pred in [("ALL", lambda r: True),
                                   ("CORR", lambda r: r["is_corridor"]),
                                   ("NON-C", lambda r: not r["is_corridor"])]:
            subset = [r for r in all_results if pred(r)]
            if not subset:
                continue
            w = l = t = 0
            for r in subset:
                a = r["strategies"][cmp][-1] if r["strategies"][cmp] else 0
                b = r["strategies"]["greedy"][-1] if r["strategies"]["greedy"] else 0
                if a > b: w += 1
                elif a < b: l += 1
                else: t += 1
            n = w + l + t
            print(f"  {cmp:<25} [{subset_name:<5}] W={w:>2} L={l:>2} T={t:>2} "
                  f"(win={w/n:.0%})" if n else "", flush=True)

    # Step-by-step for corridor functions
    corridor = [r for r in all_results if r["is_corridor"]]
    if corridor:
        print(f"\n{'=' * 70}", flush=True)
        print("CORRIDOR step-by-step", flush=True)
        print(f"{'=' * 70}", flush=True)
        all_strats = orig + new
        print(f"  {'Step':>5}", end="", flush=True)
        for s in all_strats:
            print(f" {s[:10]:>11}", end="")
        print(flush=True)
        for step in range(args.budget):
            print(f"  {step+1:>5}", end="")
            for s in all_strats:
                vals = [r["strategies"][s][step] for r in corridor
                        if r["strategies"][s] and len(r["strategies"][s]) > step]
                print(f" {mean(vals):>11.1f}", end="")
            print(flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "toy" if args.toy else "ult"
    out_path = config.RESULTS_DIR / f"head2head_v2_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"model": config.MODEL, "budget": args.budget,
                       "K": args.K, "S": args.S, "strategies": STRATEGIES},
            "results": all_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
