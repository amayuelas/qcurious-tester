"""Full strategy comparison on corridor programs.

All strategies, one experiment:
  - random: Standard gen + random pick
  - greedy: Standard gen + LLM picks best coverage
  - curiosity_greedy: Standard gen + sampling entropy (γ=0)
  - curiosity_qvalue: Standard gen + Q-value lookahead (γ=0.5)
  - diverse_random: Diverse gen + random pick
  - diverse_art: Diverse gen + most-distant pick
  - oracle: Standard gen + execute all, pick best

Usage:
    python run_corridor_test.py --budget 15 --K 5
"""

import argparse
import json
import logging
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import (
    generate_with_model, batch_generate, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.diverse_gen import generate_diverse_candidates
from curiosity_explorer.explorer.art_selection import select_most_distant
from curiosity_explorer.explorer.info_gain import (
    _build_context, estimate_output_entropy,
)
from curiosity_explorer.explorer.q_values import compute_q_values
from curiosity_explorer.benchmarks.corridor_programs import load_corridor_programs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

STRATEGIES = [
    "random", "greedy", "curiosity_greedy",
    "curiosity_qvalue_g0", "curiosity_qvalue", "curiosity_qvalue_high",
    "oracle",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Corridor programs full comparison")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blackbox", action="store_true",
                        help="Black-box mode: LLM sees only signature + history")
    parser.add_argument("--generated", type=int, default=0,
                        help="Use N generated corridor functions instead of handwritten")
    return parser.parse_args()


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
              f"Which test covers the most NEW branches? "
              f"Respond with ONLY the number.\n\nCandidates:\n{cand_list}")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _select_oracle(candidates, func_name, source, runner):
    best, bg = candidates[0], -1
    for c in candidates:
        tr = CoverageRunner(func_name, source)
        tr.cumulative_branches = set(runner.cumulative_branches)
        r = tr.run_test(c)
        if r.new_branches > bg:
            best, bg = c, r.new_branches
    return best


def run_strategy(func_name, source, strategy, budget, K, S, gamma, seed,
                 code_visible=True):
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []
    trace = []  # detailed per-step diagnostics

    use_diverse = strategy.startswith("diverse_")

    for step in range(budget):
        if use_diverse:
            candidates = generate_diverse_candidates(
                func_name, source, test_history=test_history, K=K,
                code_visible=code_visible)
        else:
            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K,
                code_visible=code_visible)

        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            trace.append({"step": step, "candidates": [], "selected": None,
                          "new_branches": 0,
                          "cumulative": runner.get_cumulative_coverage()})
            continue

        step_info = {"step": step, "candidates": candidates}

        if strategy == "random":
            selected = random.choice(candidates)

        elif strategy == "greedy":
            selected = _select_greedy(candidates, func_name, source,
                                      test_history, runner)

        elif strategy == "curiosity_greedy":
            scores = {}
            for c in candidates:
                scores[c] = estimate_output_entropy(
                    func_name, source, test_history, c, S=S,
                    code_visible=code_visible)
            selected = max(candidates, key=lambda c: scores.get(c, 0))
            step_info["entropy_scores"] = {c: round(scores[c], 3)
                                           for c in candidates}

        elif strategy.startswith("curiosity_qvalue"):
            # γ=0: same computation, only uses immediate_ig
            # γ=0.5: uses ig + 0.5*fv (default)
            # γ=0.7: stronger planning weight
            if strategy == "curiosity_qvalue_g0":
                g = 0.0
            elif strategy == "curiosity_qvalue_high":
                g = 0.7
            else:
                g = gamma
            q_results = compute_q_values(
                func_name, source, test_history, candidates,
                gamma=g, S=S, future_K=3, code_visible=code_visible)
            selected = max(candidates,
                           key=lambda c: q_results.get(c, {}).get("q_value", 0))
            step_info["q_scores"] = {
                c: q_results.get(c, {}) for c in candidates
            }

        elif strategy == "diverse_random":
            selected = random.choice(candidates)

        elif strategy == "diverse_art":
            selected = select_most_distant(candidates, test_history)

        elif strategy == "oracle":
            selected = _select_oracle(candidates, func_name, source, runner)

        else:
            selected = candidates[0]

        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

        step_info["selected"] = selected
        step_info["output"] = result.output[:100] if result.output else None
        step_info["exception"] = result.exception[:100] if result.exception else None
        step_info["new_branches"] = result.new_branches
        step_info["cumulative"] = runner.get_cumulative_coverage()
        trace.append(step_info)

    return curve, trace


def main():
    args = parse_args()
    reset_cost()

    if args.generated > 0:
        from curiosity_explorer.benchmarks.function_generator import generate_batch
        programs = generate_batch(n=args.generated, seed=args.seed)
        print(f"  Generated {len(programs)} corridor functions", flush=True)
    else:
        programs = load_corridor_programs()

    print("=" * 70, flush=True)
    mode = "BLACK-BOX" if args.blackbox else "WHITE-BOX"
    print(f"Corridor Programs — Full Strategy Comparison ({mode})", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}, S={args.S}, γ={args.gamma}",
          flush=True)
    print(f"  Programs: {list(programs.keys())}", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)
    print("=" * 70, flush=True)

    test = generate_with_model(config.MODEL, "Say 'ok'", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        desc = prog.get("description", "")

        print(f"\n[{i+1}/{len(programs)}] {key} ({func_name})", flush=True)
        print(f"  {desc}", flush=True)

        func_results = {
            "func_key": key, "func_name": func_name,
            "description": desc, "strategies": {}, "traces": {},
        }

        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {
                executor.submit(run_strategy, func_name, source, s,
                                args.budget, args.K, args.S, args.gamma,
                                args.seed,
                                code_visible=not args.blackbox): s
                for s in STRATEGIES
            }
            for future in as_completed(futures):
                s = futures[future]
                try:
                    curve, trace = future.result()
                except Exception as e:
                    log.warning(f"{s} failed on {key}: {e}")
                    curve, trace = [], []
                func_results["strategies"][s] = curve
                func_results["traces"][s] = trace
                final = curve[-1] if curve else 0
                print(f"  {s:<25} final={final}", flush=True)

        all_results.append(func_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Analysis
    def mean(v):
        return sum(v) / len(v) if v else 0

    # Summary table
    print(f"\n{'=' * 70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'Strategy':<25} {'Mean Final':>10} {'Mean @5':>10} {'Mean @10':>10}",
          flush=True)
    print(f"  {'─' * 57}", flush=True)
    for s in STRATEGIES:
        finals = [r["strategies"][s][-1] for r in all_results if r["strategies"][s]]
        at5 = [r["strategies"][s][4] for r in all_results
               if r["strategies"][s] and len(r["strategies"][s]) >= 5]
        at10 = [r["strategies"][s][9] for r in all_results
                if r["strategies"][s] and len(r["strategies"][s]) >= 10]
        print(f"  {s:<25} {mean(finals):>10.1f} {mean(at5):>10.1f} "
              f"{mean(at10):>10.1f}", flush=True)

    # Per-function table (transposed)
    print(f"\n{'=' * 70}", flush=True)
    print("PER-FUNCTION", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'Strategy':<25}", end="", flush=True)
    for r in all_results:
        print(f" {r['func_key'][:12]:>13}", end="")
    print(flush=True)
    print(f"  {'─' * 25}" + "─" * 14 * len(all_results), flush=True)
    for s in STRATEGIES:
        print(f"  {s:<25}", end="")
        for r in all_results:
            v = r["strategies"][s][-1] if r["strategies"][s] else 0
            print(f" {v:>13}", end="")
        print(flush=True)

    # Step-by-step per function
    for r in all_results:
        print(f"\n  {r['func_key']} — step-by-step:", flush=True)
        print(f"  {'Step':>5}", end="", flush=True)
        for s in STRATEGIES:
            print(f" {s[:10]:>11}", end="")
        print(flush=True)
        budget = args.budget
        for step in range(budget):
            print(f"  {step+1:>5}", end="")
            for s in STRATEGIES:
                v = r["strategies"][s]
                print(f" {v[step] if step < len(v) else 0:>11}", end="")
            print(flush=True)

    # Win rates
    print(f"\n{'=' * 70}", flush=True)
    print("WIN RATES vs GREEDY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for cmp in STRATEGIES:
        if cmp == "greedy":
            continue
        w = l = t = 0
        for r in all_results:
            a = r["strategies"][cmp][-1] if r["strategies"][cmp] else 0
            b = r["strategies"]["greedy"][-1] if r["strategies"]["greedy"] else 0
            if a > b: w += 1
            elif a < b: l += 1
            else: t += 1
        n = w + l + t
        if n:
            print(f"  {cmp:<25} W={w} L={l} T={t} (win={w/n:.0%})", flush=True)

    # Q-value vs curiosity_greedy (the key comparison)
    print(f"\n{'=' * 70}", flush=True)
    print("Q-VALUE vs GREEDY CURIOSITY (γ=0.5 vs γ=0)", flush=True)
    print(f"{'=' * 70}", flush=True)
    for r in all_results:
        q = r["strategies"]["curiosity_qvalue"][-1] if r["strategies"]["curiosity_qvalue"] else 0
        g = r["strategies"]["curiosity_greedy"][-1] if r["strategies"]["curiosity_greedy"] else 0
        diff = q - g
        winner = "Q-VALUE" if diff > 0 else "GREEDY" if diff < 0 else "TIE"
        print(f"  {r['func_key']:<20} qvalue={q:>4} greedy={g:>4} diff={diff:>+4} → {winner}",
              flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "blackbox" if args.blackbox else "whitebox"
    with open(config.RESULTS_DIR / f"corridor_test_{tag}.json", "w") as f:
        json.dump({
            "config": {"model": config.MODEL, "budget": args.budget,
                       "K": args.K, "S": args.S, "gamma": args.gamma},
            "results": all_results,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"Saved to results/corridor_test_results.json", flush=True)


if __name__ == "__main__":
    main()
