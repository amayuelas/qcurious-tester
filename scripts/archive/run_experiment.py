"""Main experiment driver for curiosity-guided code exploration.

Usage:
    pip install -r requirements.txt
    # Set GEMINI_API_KEY in .env
    python run_experiment.py                          # toy programs (default)
    python run_experiment.py --benchmark cruxeval --max-functions 10
    python run_experiment.py --benchmark ult --max-functions 5 --min-complexity 10
"""

import argparse
import json
import logging
import sys
import time

import config
from curiosity_explorer.llm import llm_generate, cache_stats, get_cost, reconfigure
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.benchmarks import load_benchmark
from curiosity_explorer.explorer.baselines import (
    run_random_strategy,
    run_greedy_coverage_strategy,
)
from curiosity_explorer.explorer.curiosity_search import run_curiosity_strategy
from curiosity_explorer.analysis.calibration import analyze_calibration
from curiosity_explorer.analysis.corridor_analysis import classify_corridor_structure
from curiosity_explorer.analysis.plotting import (
    print_comparison_table,
    print_coverage_curves,
    print_calibration_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curiosity-guided code exploration experiment"
    )
    parser.add_argument(
        "--model", default=None,
        help=f"LLM model to use (default: {config.MODEL})",
    )
    parser.add_argument(
        "--benchmark", default="toy",
        choices=["toy", "cruxeval", "ult"],
        help="Benchmark to run (default: toy)",
    )
    parser.add_argument(
        "--max-functions", type=int, default=None,
        help="Max number of functions to load from the benchmark",
    )
    parser.add_argument(
        "--min-complexity", type=int, default=None,
        help="Min cyclomatic complexity filter (ULT only)",
    )
    parser.add_argument(
        "--max-complexity", type=int, default=None,
        help="Max cyclomatic complexity filter (ULT only)",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle functions before selecting",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=f"Number of exploration steps per function (default: {config.BUDGET})",
    )
    parser.add_argument(
        "--K", type=int, default=None,
        help=f"Number of candidates per step (default: {config.K})",
    )
    parser.add_argument(
        "--S", type=int, default=None,
        help=f"Number of survivors per step (default: {config.S})",
    )
    return parser.parse_args()


def run_experiment(programs=None, budget=None, K=None, S=None):
    """Run the full exploration experiment on a set of programs."""
    from curiosity_explorer.benchmarks.toy_programs import TOY_PROGRAMS
    programs = programs or TOY_PROGRAMS
    budget = budget or config.BUDGET
    K = K or config.K
    S = S or config.S

    print("=" * 70)
    print("Curiosity-Guided Code Exploration")
    print(f"  Programs: {len(programs)}, Budget: {budget}, K: {K}, S: {S}")
    print("=" * 70)

    if config.MODEL.startswith("mistral"):
        if not config.MISTRAL_API_KEY:
            print("ERROR: Set MISTRAL_API_KEY in .env or as environment variable")
            sys.exit(1)
    else:
        if not config.GEMINI_API_KEY:
            print("ERROR: Set GEMINI_API_KEY in .env or as environment variable")
            sys.exit(1)

    # Test connectivity
    print("\n[1/5] Testing LLM connectivity...")
    test_response = llm_generate("Say 'hello' and nothing else.")
    if not test_response:
        print("ERROR: LLM API not responding")
        sys.exit(1)
    print(f"  LLM responded: {test_response[:50]}")

    all_results = {}
    start_time = time.time()

    for prog_name, prog_info in programs.items():
        func_name = prog_info["func_name"]
        source = prog_info["source"]

        print(f"\n{'='*70}")
        print(f"[2/5] Testing: {prog_name} ({func_name})")
        print(f"  Description: {prog_info['description']}")

        # Corridor analysis
        structure = classify_corridor_structure(source)
        print(f"  Structure: {structure['structure']} "
              f"(complexity={structure['cyclomatic_complexity']}, "
              f"nesting={structure['max_nesting_depth']})")

        prog_results = {
            "corridor_depth": prog_info.get("corridor_depth", 0),
            "structure": structure,
        }

        # --- Random strategy ---
        print(f"\n  Running: Random (no feedback)")
        runner = CoverageRunner(func_name, source)
        try:
            random_steps = run_random_strategy(func_name, source, runner, budget)
        except Exception as e:
            log.error(f"Random strategy failed: {e}")
            random_steps = []
        random_curve = [s["cumulative"] for s in random_steps]
        prog_results["random_steps"] = random_steps
        prog_results["random_curve"] = random_curve
        prog_results["random_final"] = random_curve[-1] if random_curve else 0
        print(f"  Final coverage (arcs): {prog_results['random_final']}")

        # --- Greedy coverage strategy ---
        print(f"\n  Running: Greedy coverage feedback")
        runner = CoverageRunner(func_name, source)
        try:
            greedy_steps = run_greedy_coverage_strategy(func_name, source, runner, budget)
        except Exception as e:
            log.error(f"Greedy strategy failed: {e}")
            greedy_steps = []
        greedy_curve = [s["cumulative"] for s in greedy_steps]
        prog_results["greedy_steps"] = greedy_steps
        prog_results["greedy_curve"] = greedy_curve
        prog_results["greedy_final"] = greedy_curve[-1] if greedy_curve else 0
        print(f"  Final coverage (arcs): {prog_results['greedy_final']}")

        # --- Curiosity strategy (code visible) ---
        print(f"\n  Running: Curiosity-guided (K={K}, S={S})")
        runner = CoverageRunner(func_name, source)
        try:
            curiosity_steps, diagnostics = run_curiosity_strategy(
                func_name, source, runner, budget, K=K, S=S,
                code_visible=True
            )
        except Exception as e:
            log.error(f"Curiosity strategy failed: {e}")
            curiosity_steps, diagnostics = [], []
        curiosity_curve = [s["cumulative"] for s in curiosity_steps]
        prog_results["curiosity_steps"] = curiosity_steps
        prog_results["curiosity_curve"] = curiosity_curve
        prog_results["curiosity_final"] = curiosity_curve[-1] if curiosity_curve else 0
        prog_results["diagnostics"] = diagnostics
        print(f"  Final coverage (arcs): {prog_results['curiosity_final']}")

        # --- Curiosity strategy (code hidden) ---
        print(f"\n  Running: Curiosity-blind (K={K}, S={S}, code_visible=False)")
        runner = CoverageRunner(func_name, source)
        try:
            blind_steps, blind_diagnostics = run_curiosity_strategy(
                func_name, source, runner, budget, K=K, S=S,
                code_visible=False
            )
        except Exception as e:
            log.error(f"Curiosity-blind strategy failed: {e}")
            blind_steps, blind_diagnostics = [], []
        blind_curve = [s["cumulative"] for s in blind_steps]
        prog_results["blind_steps"] = blind_steps
        prog_results["blind_curve"] = blind_curve
        prog_results["blind_final"] = blind_curve[-1] if blind_curve else 0
        prog_results["blind_diagnostics"] = blind_diagnostics
        print(f"  Final coverage (arcs): {prog_results['blind_final']}")

        # --- Calibration (code-visible curiosity) ---
        calibration = analyze_calibration(diagnostics)
        prog_results["calibration"] = calibration
        blind_calibration = analyze_calibration(blind_diagnostics)
        prog_results["blind_calibration"] = blind_calibration
        print(f"\n  Calibration (visible):")
        for k, v in calibration.items():
            print(f"    {k}: {v}")
        print(f"  Calibration (blind):")
        for k, v in blind_calibration.items():
            print(f"    {k}: {v}")

        all_results[prog_name] = prog_results

    elapsed = time.time() - start_time

    # --- Summary ---
    print("\n" + "=" * 70)
    print("[3/5] SUMMARY")
    print("=" * 70)
    print_comparison_table(all_results)
    print_coverage_curves(all_results)
    print_calibration_summary(all_results)

    # --- Go/No-Go ---
    print("\n" + "=" * 70)
    print("[4/5] GO / NO-GO ASSESSMENT")
    print("=" * 70)
    _assess(all_results)

    # --- Save ---
    print(f"\n[5/5] Saving results...")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = config.MODEL.replace("/", "_")
    output_path = config.RESULTS_DIR / f"experiment_{model_tag}.json"
    cost = get_cost()
    _save_results(all_results, output_path, cost=cost)
    print(f"  Results saved to {output_path}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Cache stats: {cache_stats()}")
    print(f"\n  Cost ({cost['model']}):")
    print(f"    API calls:     {cost['api_calls']}")
    print(f"    Input tokens:  {cost['input_tokens']:,}")
    print(f"    Output tokens: {cost['output_tokens']:,}")
    print(f"    Input cost:    ${cost['input_cost_usd']:.4f}")
    print(f"    Output cost:   ${cost['output_cost_usd']:.4f}")
    print(f"    Total cost:    ${cost['total_cost_usd']:.4f}")
    print("\nDone.")


def _assess(results):
    """Go/No-Go assessment."""
    issues = []
    signals = []

    corridor_progs = [k for k, v in results.items() if v["corridor_depth"] > 0]
    for prog in corridor_progs:
        res = results[prog]
        if res["curiosity_final"] > res["random_final"]:
            signals.append(f"✓ Curiosity > Random on {prog} "
                         f"({res['curiosity_final']:.0f} vs {res['random_final']:.0f})")
        else:
            issues.append(f"✗ Curiosity ≤ Random on {prog} "
                        f"({res['curiosity_final']:.0f} vs {res['random_final']:.0f})")

    for prog in corridor_progs:
        res = results[prog]
        if res["curiosity_final"] > res["greedy_final"]:
            signals.append(f"✓ Curiosity > Greedy on {prog} "
                         f"({res['curiosity_final']:.0f} vs {res['greedy_final']:.0f})")
        else:
            issues.append(f"? Curiosity ≤ Greedy on {prog} "
                        f"({res['curiosity_final']:.0f} vs {res['greedy_final']:.0f})")

    for prog_name, res in results.items():
        cal = res["calibration"]
        if cal.get("spearman_correlation", 0) > 0.1:
            signals.append(f"✓ Entropy-coverage ρ={cal['spearman_correlation']} on {prog_name}")
        elif cal.get("spearman_correlation", 0) > -0.1:
            issues.append(f"? Weak ρ on {prog_name}: {cal.get('spearman_correlation', 'N/A')}")
        else:
            issues.append(f"✗ Negative ρ on {prog_name}: {cal['spearman_correlation']}")

    flat_progs = [k for k, v in results.items() if v["corridor_depth"] == 0]
    for prog in flat_progs:
        res = results[prog]
        diff = abs(res["curiosity_final"] - res["greedy_final"])
        if diff < 3:
            signals.append(f"✓ Curiosity ≈ Greedy on flat {prog} (as expected)")
        elif res["curiosity_final"] > res["greedy_final"]:
            signals.append(f"✓ Curiosity > Greedy on flat {prog} (bonus)")
        else:
            issues.append(f"? Curiosity < Greedy on flat {prog}")

    # Check 5: Does blind curiosity compete with code-visible greedy?
    for prog in corridor_progs:
        res = results[prog]
        blind = res.get("blind_final", 0)
        greedy = res["greedy_final"]
        if blind > greedy:
            signals.append(f"✓ Blind > Greedy on {prog} "
                         f"({blind:.0f} vs {greedy:.0f})")
        elif abs(blind - greedy) < 3:
            signals.append(f"✓ Blind ≈ Greedy on {prog} "
                         f"({blind:.0f} vs {greedy:.0f})")
        else:
            issues.append(f"? Blind < Greedy on {prog} "
                        f"({blind:.0f} vs {greedy:.0f})")

    print("\nPositive signals:")
    for s in signals:
        print(f"  {s}")
    print("\nIssues / concerns:")
    for i in issues:
        print(f"  {i}")

    critical = [i for i in issues if i.startswith("✗")]
    if len(critical) == 0 and len(signals) >= 3:
        print("\nVERDICT: ✅ GO — proceed to Phase 1")
    elif len(critical) <= 1:
        print("\nVERDICT: ⚠️  INVESTIGATE — mixed signals")
    else:
        print("\nVERDICT: 🛑 STOP — fundamental issues detected")


def _save_results(results, path, cost=None):
    """Serialize results to JSON, including per-step details and diagnostics."""
    serializable = {}
    if cost:
        serializable["_experiment_cost"] = cost
    for k, v in results.items():
        entry = {
            "corridor_depth": v["corridor_depth"],
            "structure": v.get("structure", {}),
            "random_final": v["random_final"],
            "greedy_final": v["greedy_final"],
            "curiosity_final": v["curiosity_final"],
            "random_curve": v["random_curve"],
            "greedy_curve": v["greedy_curve"],
            "curiosity_curve": v["curiosity_curve"],
            "calibration": v["calibration"],
            "blind_final": v.get("blind_final", 0),
            "blind_curve": v.get("blind_curve", []),
            "blind_calibration": v.get("blind_calibration", {}),
            "diagnostics": v.get("diagnostics", []),
            "blind_diagnostics": v.get("blind_diagnostics", []),
        }
        # Per-step details (filter non-serializable fields)
        for strategy in ["random", "greedy", "curiosity", "blind"]:
            steps_key = f"{strategy}_steps"
            if steps_key in v:
                entry[steps_key] = [
                    {k2: v2 for k2, v2 in step.items()
                     if k2 != "all_candidates"}
                    for step in v[steps_key]
                ]
        serializable[k] = entry

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    args = parse_args()

    # Apply model override before any LLM calls
    if args.model:
        config.MODEL = args.model
        reconfigure()

    # Build loader kwargs
    loader_kwargs = {}
    if args.max_functions is not None:
        loader_kwargs["max_functions"] = args.max_functions
    if args.shuffle:
        loader_kwargs["shuffle"] = True
    if args.benchmark == "ult":
        if args.min_complexity is not None:
            loader_kwargs["min_complexity"] = args.min_complexity
        if args.max_complexity is not None:
            loader_kwargs["max_complexity"] = args.max_complexity

    programs = load_benchmark(args.benchmark, **loader_kwargs)
    print(f"Loaded {len(programs)} programs from '{args.benchmark}' benchmark")

    run_experiment(
        programs=programs,
        budget=args.budget,
        K=args.K,
        S=args.S,
    )


if __name__ == "__main__":
    main()
