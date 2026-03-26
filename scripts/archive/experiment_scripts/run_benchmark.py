"""Benchmark runner for cryptographic corridor functions.

Usage:
    python run_benchmark.py --pilot                     # 3 funcs × 2 seeds × 4 strategies
    python run_benchmark.py --n-per-level 10 --seeds 3  # full run
    python run_benchmark.py --strategies random reflective_qvalue  # specific strategies
"""

import argparse
import json
import logging
import statistics
import time

import config
from curiosity_explorer.llm import generate_with_model, get_cost, reset_cost
from curiosity_explorer.benchmarks.cryptographic_corridors import (
    generate_benchmark_suite, DIFFICULTY_LEVELS,
)
from curiosity_explorer.benchmarks.benchmark_runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

ALL_STRATEGIES = [
    "random", "greedy", "curiosity_entropy",
    "curiosity_qvalue", "reflective_qvalue", "oracle",
]

PILOT_STRATEGIES = ["random", "greedy", "curiosity_qvalue", "reflective_qvalue"]


def parse_args():
    parser = argparse.ArgumentParser(description="Cryptographic corridor benchmark")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot run: 1 func/level × 2 seeds × 4 strategies")
    parser.add_argument("--n-per-level", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def compute_aucc(curve, budget):
    """Area Under Coverage Curve, normalized by budget."""
    if not curve:
        return 0.0
    return sum(curve) / budget


def main():
    args = parse_args()
    reset_cost()

    if args.pilot:
        n_per_level = 1
        seed_list = [42, 123]
        strategies = args.strategies or PILOT_STRATEGIES
        args.budget = min(args.budget, 20)
    else:
        n_per_level = args.n_per_level
        seed_list = [42 + i * 111 for i in range(args.seeds)]
        strategies = args.strategies or ALL_STRATEGIES

    print("=" * 70, flush=True)
    print("Cryptographic Corridor Benchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Functions: {n_per_level} per level × 3 levels = {n_per_level * 3}",
          flush=True)
    print(f"  Seeds: {seed_list}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    n_runs = n_per_level * 3 * len(seed_list) * len(strategies)
    print(f"  Total runs: {n_runs}", flush=True)
    print("=" * 70, flush=True)

    # Connectivity
    test = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    # Generate benchmark suite
    programs = {}
    suite = generate_benchmark_suite(n_per_level, args.base_seed)
    programs.update(suite)
    print(f"  Generated {len(programs)} functions", flush=True)

    for level in DIFFICULTY_LEVELS:
        level_funcs = [k for k in programs if k.startswith(level)]
        print(f"    {level}: {len(level_funcs)} functions", flush=True)

    # Run benchmark
    start = time.time()
    results = run_benchmark(
        programs, strategies, budget=args.budget, K=args.K,
        S=args.S, gamma=args.gamma, seeds=seed_list,
    )
    elapsed = time.time() - start
    cost = get_cost()

    # === ANALYSIS ===
    def mean(v):
        return statistics.mean(v) if v else 0
    def sem(v):
        return statistics.stdev(v) / len(v)**0.5 if len(v) > 1 else 0

    # Group results
    by_strategy = {}
    by_strategy_level = {}
    for r in results:
        s = r.strategy
        level = r.func_key.split("_")[0]
        final = r.coverage_curve[-1] if r.coverage_curve else 0
        aucc = compute_aucc(r.coverage_curve, args.budget)

        by_strategy.setdefault(s, []).append({"final": final, "aucc": aucc,
                                              "curve": r.coverage_curve})
        by_strategy_level.setdefault((s, level), []).append(
            {"final": final, "aucc": aucc, "curve": r.coverage_curve,
             "gates": r.gate_passage_steps})

    # Overall summary
    print(f"\n{'=' * 70}", flush=True)
    print("OVERALL RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'Strategy':<25} {'Mean Final':>10} {'±SE':>8} {'Mean AUCC':>10}",
          flush=True)
    print(f"  {'─' * 55}", flush=True)
    for s in strategies:
        data = by_strategy.get(s, [])
        finals = [d["final"] for d in data]
        auccs = [d["aucc"] for d in data]
        print(f"  {s:<25} {mean(finals):>10.1f} {sem(finals):>7.1f} "
              f"{mean(auccs):>10.1f}", flush=True)

    # Per difficulty level
    for level in ["easy", "medium", "hard"]:
        print(f"\n{'=' * 70}", flush=True)
        print(f"{level.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  {'Strategy':<25} {'Mean Final':>10} {'±SE':>8} {'Mean AUCC':>10}",
              flush=True)
        print(f"  {'─' * 55}", flush=True)
        for s in strategies:
            data = by_strategy_level.get((s, level), [])
            finals = [d["final"] for d in data]
            auccs = [d["aucc"] for d in data]
            if finals:
                print(f"  {s:<25} {mean(finals):>10.1f} {sem(finals):>7.1f} "
                      f"{mean(auccs):>10.1f}", flush=True)

    # Paired comparison: each strategy vs random
    print(f"\n{'=' * 70}", flush=True)
    print("PAIRED COMPARISON vs RANDOM", flush=True)
    print(f"{'=' * 70}", flush=True)

    random_results = {}
    for r in results:
        if r.strategy == "random":
            random_results[(r.func_key, r.seed)] = (
                r.coverage_curve[-1] if r.coverage_curve else 0
            )

    for s in strategies:
        if s == "random":
            continue
        deltas = []
        for r in results:
            if r.strategy == s:
                rand_final = random_results.get((r.func_key, r.seed), 0)
                my_final = r.coverage_curve[-1] if r.coverage_curve else 0
                deltas.append(my_final - rand_final)

        if not deltas:
            continue

        md = mean(deltas)
        se = sem(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        ties = sum(1 for d in deltas if d == 0)

        # Paired t-test
        from scipy import stats
        if len(deltas) > 1 and statistics.stdev(deltas) > 0:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            sig = "*" if p_val < 0.05 else ""
        else:
            p_val = 1.0
            sig = ""

        print(f"  {s:<25} Δ={md:>+6.1f} ±{se:.1f} "
              f"W={wins} L={losses} T={ties} "
              f"p={p_val:.4f} {sig}", flush=True)

        # Per difficulty
        for level in ["easy", "medium", "hard"]:
            level_deltas = [d for r, d in zip(
                [r for r in results if r.strategy == s],
                deltas
            ) if r.func_key.startswith(level)]
            if level_deltas:
                lmd = mean(level_deltas)
                lw = sum(1 for d in level_deltas if d > 0)
                ll = sum(1 for d in level_deltas if d < 0)
                print(f"    [{level:<6}] Δ={lmd:>+6.1f} W={lw} L={ll}",
                      flush=True)

    # Cost
    print(f"\n{'=' * 70}", flush=True)
    print(f"  Cost: ${cost['total_cost_usd']:.4f}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "config": {
            "model": config.MODEL, "budget": args.budget, "K": args.K,
            "S": args.S, "gamma": args.gamma, "seeds": seed_list,
            "strategies": strategies, "n_per_level": n_per_level,
        },
        "results": [
            {"func_key": r.func_key, "strategy": r.strategy, "seed": r.seed,
             "coverage_curve": r.coverage_curve,
             "gate_passage_steps": r.gate_passage_steps,
             "elapsed": r.elapsed_seconds}
            for r in results
        ],
        "cost": cost,
        "elapsed_seconds": round(elapsed, 1),
    }
    tag = "pilot" if args.pilot else "full"
    out_path = config.RESULTS_DIR / f"benchmark_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
