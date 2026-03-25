"""Test forked stdlib functions in white-box and black-box modes.

Usage:
    python run_forked_test.py                    # white-box
    python run_forked_test.py --blackbox         # black-box
    python run_forked_test.py --strategies random reflective_qvalue
"""

import argparse
import json
import logging
import time

import config
from curiosity_explorer.llm import generate_with_model, get_cost, reset_cost
from curiosity_explorer.benchmarks.forked_stdlib import load_forked_programs
from curiosity_explorer.benchmarks.benchmark_runner import run_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

ALL_STRATEGIES = ["random", "greedy", "curiosity_qvalue", "reflective_qvalue"]


def parse_args():
    parser = argparse.ArgumentParser(description="Forked stdlib test")
    parser.add_argument("--blackbox", action="store_true")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--S", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--functions", nargs="+", default=None,
                        help="Which functions to test (default: shlex_modified htmlparser_modified)")
    return parser.parse_args()


def main():
    args = parse_args()
    reset_cost()

    mode = "BLACK-BOX" if args.blackbox else "WHITE-BOX"
    strategies = args.strategies or ALL_STRATEGIES
    func_keys = args.functions or ["shlex_modified", "htmlparser_modified"]

    programs = load_forked_programs()
    test_funcs = {k: programs[k] for k in func_keys if k in programs}

    print("=" * 70, flush=True)
    print(f"Forked Stdlib — {mode}", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model: {config.MODEL}", flush=True)
    print(f"  Mode: {mode}", flush=True)
    print(f"  Budget: {args.budget}, K={args.K}", flush=True)
    print(f"  Functions: {list(test_funcs.keys())}", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    print("=" * 70, flush=True)

    test = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"  Connectivity: {'OK' if test else 'FAILED'}", flush=True)
    if not test:
        return

    start = time.time()
    all_results = []

    for key, prog in test_funcs.items():
        print(f"\n{key} ({prog['metadata']['description']})", flush=True)

        func_results = {"func_key": key, "strategies": {}}

        for s in strategies:
            curve, gates = run_single(
                prog["func_name"], prog["source"], s,
                budget=args.budget, K=args.K, S=args.S,
                seed=args.seed, code_visible=not args.blackbox,
            )
            final = curve[-1] if curve else 0
            func_results["strategies"][s] = {
                "curve": curve, "gates": gates, "final": final,
            }
            print(f"  {s:<25} final={final}", flush=True)

        all_results.append(func_results)

    elapsed = time.time() - start
    cost = get_cost()

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"SUMMARY ({mode})", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Function':<22}", end="", flush=True)
    for s in strategies:
        print(f" {s[:12]:>13}", end="")
    print(flush=True)
    for r in all_results:
        print(f"{r['func_key']:<22}", end="")
        for s in strategies:
            v = r["strategies"].get(s, {}).get("final", 0)
            print(f" {v:>13}", end="")
        print(flush=True)

    # Mean
    print(f"{'MEAN':<22}", end="", flush=True)
    for s in strategies:
        vals = [r["strategies"].get(s, {}).get("final", 0) for r in all_results]
        mean = sum(vals) / len(vals) if vals else 0
        print(f" {mean:>13.1f}", end="")
    print(flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "bb" if args.blackbox else "wb"
    with open(config.RESULTS_DIR / f"forked_stdlib_{tag}.json", "w") as f:
        json.dump({"mode": mode, "config": {"budget": args.budget, "K": args.K},
                    "results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print(f"  Saved", flush=True)


if __name__ == "__main__":
    main()
