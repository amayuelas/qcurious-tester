"""Phase 1 calibration: does the information gain estimator predict coverage gain?

Supports two estimators:
  - "output_entropy": LLM predicts function output, measure Shannon entropy
  - "coverage_disagreement": LLM predicts whether test hits new branches, measure disagreement

For each function, at exploration states [0, 5, 10]:
  - Generate K candidate test inputs
  - Score each candidate with the estimator
  - Actually run each to measure real coverage gain (new branches)
  - Record (score, actual_new_branches) pairs

Then compute Spearman correlation overall and stratified by complexity/state.

Usage:
    # Output entropy (original), S=20, 20+ complexity only
    python run_calibration.py --estimator output_entropy --S 20 --min-complexity 20

    # Coverage disagreement (new), S=20, 20+ complexity only
    python run_calibration.py --estimator coverage_disagreement --S 20 --min-complexity 20

    # Both estimators side by side
    python run_calibration.py --estimator both --S 20 --min-complexity 20
"""

import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

import config
from curiosity_explorer.llm import (
    llm_generate, reconfigure, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.benchmarks import load_benchmark
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.info_gain import (
    estimate_output_entropy,
    estimate_coverage_disagreement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CALIBRATION_STATES = [0, 5, 10]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 calibration: estimator vs coverage gain"
    )
    parser.add_argument(
        "--model", default=None,
        help=f"LLM model to use (default: {config.MODEL})",
    )
    parser.add_argument(
        "--estimator", default="both",
        choices=["output_entropy", "coverage_disagreement", "both"],
        help="Which estimator to calibrate (default: both)",
    )
    parser.add_argument(
        "--num-functions", type=int, default=100,
        help="Number of functions to calibrate on (default: 100)",
    )
    parser.add_argument(
        "--K", type=int, default=10,
        help="Number of candidate tests per state (default: 10)",
    )
    parser.add_argument(
        "--S", type=int, default=20,
        help="Number of prediction samples (default: 20)",
    )
    parser.add_argument(
        "--min-complexity", type=int, default=None,
        help="Minimum cyclomatic complexity filter",
    )
    parser.add_argument(
        "--max-complexity", type=int, default=None,
        help="Maximum cyclomatic complexity filter",
    )
    parser.add_argument(
        "--benchmark", default="ult",
        choices=["ult", "cruxeval"],
        help="Benchmark to calibrate on (default: ult)",
    )
    return parser.parse_args()


def build_up_state(func_name, source, runner, test_history, target_count, K=5):
    """Run tests to build exploration state to target_count prior tests."""
    while len(test_history) < target_count:
        candidates = generate_test_candidates(
            func_name, source, test_history=test_history, K=K
        )
        if not candidates:
            break
        for cand in candidates:
            result = runner.run_test(cand)
            test_history.append((cand, result))
            break


def _score_candidate(func_name, source, test_history, runner_branches, cand, S,
                     estimators):
    """Score a single candidate with one or both estimators + actual coverage."""
    result_dict = {"test": cand}

    if "output_entropy" in estimators:
        result_dict["output_entropy"] = estimate_output_entropy(
            func_name, source, test_history, cand, S=S
        )

    if "coverage_disagreement" in estimators:
        result_dict["coverage_disagreement"] = estimate_coverage_disagreement(
            func_name, source, test_history, cand,
            cumulative_arcs=len(runner_branches), S=S
        )

    # Actual coverage gain
    temp_runner = CoverageRunner(func_name, source)
    temp_runner.cumulative_branches = set(runner_branches)
    result = temp_runner.run_test(cand)
    result_dict["new_branches"] = result.new_branches
    result_dict["output"] = result.output
    result_dict["exception"] = result.exception

    return result_dict


def calibrate_at_state(func_name, source, runner, test_history, K, S, estimators):
    """Generate K candidates, score with estimators, run each, return data."""
    candidates = generate_test_candidates(
        func_name, source, test_history=test_history, K=K
    )
    if not candidates:
        return []

    data_points = []
    with ThreadPoolExecutor(max_workers=min(len(candidates), 10)) as executor:
        futures = {
            executor.submit(
                _score_candidate, func_name, source, test_history,
                runner.cumulative_branches, cand, S, estimators
            ): cand
            for cand in candidates
        }
        for future in as_completed(futures):
            try:
                data_points.append(future.result())
            except Exception as e:
                log.warning(f"Candidate scoring failed: {e}")

    return data_points


def run_calibration(programs, K, S, estimators):
    """Run calibration across all programs and states."""
    all_data = []
    total = len(programs)

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        complexity = prog.get("metadata", {}).get("cyclomatic_complexity", 0)

        print(f"\n[{i+1}/{total}] {key} ({func_name}, complexity={complexity})")

        runner = CoverageRunner(func_name, source)
        test_history = []

        for state in CALIBRATION_STATES:
            build_up_state(func_name, source, runner, test_history, state, K=3)
            actual_state = len(test_history)

            print(f"  State {state} (actual={actual_state}, "
                  f"arcs={runner.get_cumulative_coverage()}): "
                  f"{K} candidates...")

            points = calibrate_at_state(
                func_name, source, runner, test_history, K=K, S=S,
                estimators=estimators
            )

            for pt in points:
                pt["state"] = state
                pt["complexity"] = complexity
                pt["func_key"] = key
                pt["func_name"] = func_name
                all_data.append(pt)

            # Print per-state summary for each estimator
            gains = [p["new_branches"] for p in points]
            for est in estimators:
                scores = [p.get(est, 0) for p in points]
                if len(points) >= 3 and len(set(scores)) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", stats.ConstantInputWarning)
                        rho, _ = stats.spearmanr(scores, gains)
                    tag = "OE" if est == "output_entropy" else "CD"
                    print(f"    [{tag}] rho={rho:.3f} | "
                          f"mean_score={sum(scores)/len(scores):.2f} | "
                          f"mean_gain={sum(gains)/len(gains):.1f}")

    return all_data


def analyze_results(all_data, estimator_name):
    """Compute correlations for a single estimator."""
    results = {}
    scores = [d.get(estimator_name, 0) for d in all_data]
    gains = [d["new_branches"] for d in all_data]

    if len(all_data) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho, pval = stats.spearmanr(scores, gains)
        results["overall"] = {
            "n": len(all_data),
            "spearman_rho": round(rho, 4) if not (rho != rho) else 0,
            "p_value": round(pval, 6) if not (pval != pval) else 1,
            "mean_score": round(sum(scores) / len(scores), 4),
            "mean_gain": round(sum(gains) / len(gains), 4),
        }

    # By complexity
    buckets = {
        "10-15": lambda c: 10 <= c < 15,
        "15-20": lambda c: 15 <= c < 20,
        "20+": lambda c: c >= 20,
    }
    results["by_complexity"] = {}
    for label, pred in buckets.items():
        subset = [d for d in all_data if pred(d["complexity"])]
        if len(subset) >= 3:
            e = [d.get(estimator_name, 0) for d in subset]
            g = [d["new_branches"] for d in subset]
            if len(set(e)) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", stats.ConstantInputWarning)
                    rho, pval = stats.spearmanr(e, g)
            else:
                rho, pval = 0, 1
            results["by_complexity"][label] = {
                "n": len(subset),
                "n_functions": len(set(d["func_key"] for d in subset)),
                "spearman_rho": round(rho, 4) if not (rho != rho) else 0,
                "p_value": round(pval, 6) if not (pval != pval) else 1,
                "mean_score": round(sum(e) / len(e), 4),
                "mean_gain": round(sum(g) / len(g), 4),
            }

    # By state
    results["by_state"] = {}
    for state in CALIBRATION_STATES:
        subset = [d for d in all_data if d["state"] == state]
        if len(subset) >= 3:
            e = [d.get(estimator_name, 0) for d in subset]
            g = [d["new_branches"] for d in subset]
            if len(set(e)) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", stats.ConstantInputWarning)
                    rho, pval = stats.spearmanr(e, g)
            else:
                rho, pval = 0, 1
            results["by_state"][f"state_{state}"] = {
                "n": len(subset),
                "spearman_rho": round(rho, 4) if not (rho != rho) else 0,
                "p_value": round(pval, 6) if not (pval != pval) else 1,
                "mean_score": round(sum(e) / len(e), 4),
                "mean_gain": round(sum(g) / len(g), 4),
            }

    return results


def print_report(estimator_name, analysis, cost):
    """Print formatted calibration report for one estimator."""
    tag = "Output Entropy" if estimator_name == "output_entropy" else "Coverage Disagreement"
    print(f"\n{'=' * 70}")
    print(f"CALIBRATION REPORT: {tag}")
    print(f"{'=' * 70}")

    ov = analysis.get("overall", {})
    rho = ov.get("spearman_rho", 0)
    print(f"\nOverall: n={ov.get('n', 0)}, "
          f"Spearman rho={rho}, "
          f"p={ov.get('p_value', 'N/A')}")
    print(f"  Mean score: {ov.get('mean_score', 0):.3f}, "
          f"Mean gain: {ov.get('mean_gain', 0):.2f}")

    if analysis.get("by_complexity"):
        print("\nBy complexity:")
        for label, data in analysis["by_complexity"].items():
            print(f"  [{label}] n={data['n']} ({data['n_functions']} funcs), "
                  f"rho={data['spearman_rho']:.3f} (p={data['p_value']:.4f}), "
                  f"mean_score={data['mean_score']:.3f}, "
                  f"mean_gain={data['mean_gain']:.2f}")

    if analysis.get("by_state"):
        print("\nBy exploration state:")
        for label, data in analysis["by_state"].items():
            print(f"  [{label}] n={data['n']}, "
                  f"rho={data['spearman_rho']:.3f} (p={data['p_value']:.4f}), "
                  f"mean_score={data['mean_score']:.3f}, "
                  f"mean_gain={data['mean_gain']:.2f}")

    print(f"\n{'-' * 70}")
    if rho > 0.15:
        print(f"VERDICT: GO (rho={rho:.3f} > 0.15)")
    elif rho > 0:
        print(f"VERDICT: PIVOT (rho={rho:.3f} — weak)")
    else:
        print(f"VERDICT: STOP (rho={rho:.3f} — does not predict gain)")


def main():
    args = parse_args()

    if args.model:
        config.MODEL = args.model
        reconfigure()

    reset_cost()

    # Determine which estimators to run
    if args.estimator == "both":
        estimators = ["output_entropy", "coverage_disagreement"]
    else:
        estimators = [args.estimator]

    print("=" * 70)
    print(f"Phase 1 Calibration")
    print(f"  Model: {config.MODEL}")
    print(f"  Estimators: {', '.join(estimators)}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Functions: {args.num_functions}")
    print(f"  K={args.K} candidates, S={args.S} prediction samples")
    print(f"  Complexity filter: min={args.min_complexity}, max={args.max_complexity}")
    print(f"  States: {CALIBRATION_STATES}")
    print("=" * 70)

    # Test connectivity
    print("\nTesting LLM connectivity...")
    test = llm_generate("Say 'ok'")
    if not test:
        print(f"ERROR: LLM not responding ({config.MODEL})")
        return
    print(f"  OK: {test[:30]}")

    # Load functions
    if args.min_complexity is not None or args.max_complexity is not None:
        # Single complexity range
        programs = load_benchmark(
            args.benchmark,
            min_complexity=args.min_complexity,
            max_complexity=args.max_complexity,
            max_functions=args.num_functions,
            shuffle=True,
            seed=42,
        )
        print(f"  Loaded {len(programs)} functions "
              f"(complexity {args.min_complexity or '?'}-{args.max_complexity or '?'})")
    else:
        # Stratified loading
        per_bucket = args.num_functions // 3
        remainder = args.num_functions - per_bucket * 3
        programs = {}
        for label, lo, hi, count in [
            ("10-15", 10, 15, per_bucket),
            ("15-20", 15, 20, per_bucket),
            ("20+", 20, None, per_bucket + remainder),
        ]:
            bucket = load_benchmark(
                args.benchmark,
                min_complexity=lo,
                max_complexity=hi,
                max_functions=count,
                shuffle=True,
                seed=42,
            )
            print(f"  Loaded {len(bucket)} functions for complexity {label}")
            programs.update(bucket)

    print(f"  Total: {len(programs)} functions")

    start = time.time()
    all_data = run_calibration(programs, K=args.K, S=args.S, estimators=estimators)
    elapsed = time.time() - start
    cost = get_cost()

    # Analyze and report each estimator
    all_analyses = {}
    for est in estimators:
        analysis = analyze_results(all_data, est)
        all_analyses[est] = analysis
        print_report(est, analysis, cost)

    # Comparison summary if both
    if len(estimators) == 2:
        print(f"\n{'=' * 70}")
        print("HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 70}")
        oe = all_analyses["output_entropy"].get("overall", {})
        cd = all_analyses["coverage_disagreement"].get("overall", {})
        print(f"  Output Entropy:          rho={oe.get('spearman_rho', 'N/A')}")
        print(f"  Coverage Disagreement:   rho={cd.get('spearman_rho', 'N/A')}")
        oe_rho = oe.get('spearman_rho', 0)
        cd_rho = cd.get('spearman_rho', 0)
        if cd_rho > oe_rho:
            print(f"  Winner: Coverage Disagreement (+{cd_rho - oe_rho:.3f})")
        elif oe_rho > cd_rho:
            print(f"  Winner: Output Entropy (+{oe_rho - cd_rho:.3f})")
        else:
            print(f"  Tied")

    print(f"\nCost ({cost['model']}):")
    print(f"  API calls:     {cost['api_calls']}")
    print(f"  Input tokens:  {cost['input_tokens']:,}")
    print(f"  Output tokens: {cost['output_tokens']:,}")
    print(f"  Total cost:    ${cost['total_cost_usd']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = config.MODEL.replace("/", "_")
    est_tag = args.estimator
    out_path = config.RESULTS_DIR / f"calibration_{est_tag}_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "model": config.MODEL,
                "estimators": estimators,
                "benchmark": args.benchmark,
                "num_functions": args.num_functions,
                "K": args.K,
                "S": args.S,
                "min_complexity": args.min_complexity,
                "max_complexity": args.max_complexity,
                "states": CALIBRATION_STATES,
            },
            "analyses": all_analyses,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
            "data": all_data,
        }, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
