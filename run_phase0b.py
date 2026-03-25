"""Phase 0b: Uncertainty readout diagnostic.

Tests whether LLMs have useful uncertainty about program behavior that can be
read out through methods other than single-model sampling entropy.

Estimators tested:
  A — Multi-model disagreement (3 model families)
  B — Token-level logprob entropy (via Fireworks gpt-oss-120b)
  C — Verbalized confidence (self-reported 0-100)
  D — Hybrid (weighted combination of A + C)

Setup: 30 ULT functions (20+ complexity), K=10 candidates, states [0, 5, 10].
Reuses candidate generation + coverage execution from Phase 1 infrastructure.

Usage:
    python run_phase0b.py
    python run_phase0b.py --num-functions 10 --K 5  # quick test
    python run_phase0b.py --models gemini-3-flash-preview mistral-large-latest gpt-5.4-mini
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
    generate_with_model, reconfigure, get_cost, reset_cost,
)
from curiosity_explorer.runner.coverage import CoverageRunner
from curiosity_explorer.benchmarks import load_benchmark
from curiosity_explorer.explorer.candidate_gen import generate_test_candidates
from curiosity_explorer.explorer.info_gain import score_candidate_phase0b

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CALIBRATION_STATES = [0, 5, 10]
ESTIMATOR_NAMES = ["multi_model_disagreement", "logprob_entropy",
                   "verbalized_confidence", "hybrid"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0b: Uncertainty readout diagnostic"
    )
    parser.add_argument(
        "--num-functions", type=int, default=30,
        help="Number of ULT functions to test (default: 30)",
    )
    parser.add_argument(
        "--K", type=int, default=10,
        help="Candidate tests per state (default: 10)",
    )
    parser.add_argument(
        "--min-complexity", type=int, default=20,
        help="Minimum cyclomatic complexity (default: 20)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models for ensemble (default: config.ENSEMBLE_MODELS)",
    )
    parser.add_argument(
        "--benchmark", default="ult",
        choices=["ult", "cruxeval"],
        help="Benchmark (default: ult)",
    )
    return parser.parse_args()


def build_up_state(func_name, source, runner, test_history, target_count, K=3):
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


def score_candidate_with_coverage(func_name, source, test_history,
                                  runner_branches, cand, models):
    """Score one candidate with all Phase 0b estimators + measure actual coverage."""
    # Phase 0b estimator scores
    scores = score_candidate_phase0b(
        func_name, source, test_history, cand, models=models,
    )

    # Actual coverage gain (ground truth)
    temp_runner = CoverageRunner(func_name, source)
    temp_runner.cumulative_branches = set(runner_branches)
    result = temp_runner.run_test(cand)

    scores["test"] = cand
    scores["new_branches"] = result.new_branches
    scores["output"] = result.output
    scores["exception"] = result.exception

    return scores


def run_phase0b(programs, K, models):
    """Run Phase 0b diagnostic across all programs and states."""
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
                  f"scoring {K} candidates...")

            # Generate candidates
            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K
            )
            if not candidates:
                print(f"    No candidates generated, skipping")
                continue

            # Score candidates in parallel (each candidate calls 3+ models)
            points = []
            with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as executor:
                futures = {
                    executor.submit(
                        score_candidate_with_coverage,
                        func_name, source, test_history,
                        runner.cumulative_branches, cand, models,
                    ): cand
                    for cand in candidates
                }
                for future in as_completed(futures):
                    try:
                        points.append(future.result())
                    except Exception as e:
                        log.warning(f"Scoring failed: {e}")

            for pt in points:
                pt["state"] = state
                pt["complexity"] = complexity
                pt["func_key"] = key
                pt["func_name"] = func_name
                all_data.append(pt)

            # Print per-state summary
            gains = [p["new_branches"] for p in points]
            for est in ESTIMATOR_NAMES:
                scores = [p.get(est, 0) for p in points]
                if len(points) >= 3 and len(set(scores)) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rho, _ = stats.spearmanr(scores, gains)
                    print(f"    [{est[:10]}] rho={rho:+.3f} | "
                          f"mean={sum(scores)/len(scores):.3f} | "
                          f"gain={sum(gains)/len(gains):.1f}")

    return all_data


def analyze_results(all_data, estimator_name):
    """Compute Spearman correlations for one estimator."""
    results = {}
    scores = [d.get(estimator_name, 0) for d in all_data]
    gains = [d["new_branches"] for d in all_data]

    if len(all_data) >= 3 and len(set(scores)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, pval = stats.spearmanr(scores, gains)
        results["overall"] = {
            "n": len(all_data),
            "spearman_rho": round(rho, 4) if rho == rho else 0,
            "p_value": round(pval, 6) if pval == pval else 1,
            "mean_score": round(sum(scores) / len(scores), 4),
            "mean_gain": round(sum(gains) / len(gains), 4),
        }
    else:
        results["overall"] = {
            "n": len(all_data),
            "spearman_rho": 0,
            "p_value": 1,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "mean_gain": round(sum(gains) / len(gains), 4) if gains else 0,
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
                    warnings.simplefilter("ignore")
                    rho, pval = stats.spearmanr(e, g)
            else:
                rho, pval = 0, 1
            results["by_state"][f"state_{state}"] = {
                "n": len(subset),
                "spearman_rho": round(rho, 4) if rho == rho else 0,
                "p_value": round(pval, 6) if pval == pval else 1,
                "mean_score": round(sum(e) / len(e), 4),
                "mean_gain": round(sum(g) / len(g), 4),
            }

    return results


def print_report(estimator_name, analysis):
    """Print formatted report for one estimator."""
    labels = {
        "multi_model_disagreement": "Multi-Model Disagreement (Estimator A)",
        "logprob_entropy": "Token-Level Logprob Entropy (Estimator B)",
        "verbalized_confidence": "Verbalized Confidence (Estimator C)",
        "hybrid": "Hybrid A+C (Estimator D)",
    }
    label = labels.get(estimator_name, estimator_name)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    ov = analysis.get("overall", {})
    rho = ov.get("spearman_rho", 0)
    print(f"\n  Overall: n={ov.get('n', 0)}, "
          f"rho={rho:+.4f}, "
          f"p={ov.get('p_value', 'N/A')}")
    print(f"  Mean score: {ov.get('mean_score', 0):.4f}, "
          f"Mean gain: {ov.get('mean_gain', 0):.2f}")

    if analysis.get("by_state"):
        print("\n  By exploration state:")
        for label, data in analysis["by_state"].items():
            print(f"    [{label}] n={data['n']}, "
                  f"rho={data['spearman_rho']:+.4f} "
                  f"(p={data['p_value']:.4f}), "
                  f"mean_score={data['mean_score']:.4f}, "
                  f"mean_gain={data['mean_gain']:.2f}")

    # Verdict
    print(f"\n  {'─' * 50}")
    if rho > 0.15:
        print(f"  VERDICT: GO (rho={rho:+.3f} > 0.15)")
    elif rho > 0.05:
        print(f"  VERDICT: PIVOT (rho={rho:+.3f} — weak signal)")
    else:
        print(f"  VERDICT: STOP (rho={rho:+.3f} — no useful signal)")


def main():
    args = parse_args()

    models = args.models or config.ENSEMBLE_MODELS
    reset_cost()

    print("=" * 70)
    print("Phase 0b: Uncertainty Readout Diagnostic")
    print("=" * 70)
    print(f"  Ensemble models: {models}")
    print(f"  Logprob model: {config.LOGPROB_MODEL}")
    print(f"  Verbalized confidence model: {config.MODEL}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Functions: {args.num_functions} (complexity >= {args.min_complexity})")
    print(f"  K={args.K} candidates per state")
    print(f"  States: {CALIBRATION_STATES}")
    print(f"  Estimators: {ESTIMATOR_NAMES}")
    print("=" * 70)

    # Verify all models are reachable
    print("\nTesting model connectivity...")
    all_models = list(models) + [config.LOGPROB_MODEL]
    for m in all_models:
        test = generate_with_model(m, "Say 'ok'", temperature=0.3, max_tokens=10)
        status = "OK" if test else "FAILED"
        print(f"  {m}: {status}" + (f" ({test[:20]})" if test else ""))
        if not test:
            print(f"\n  ERROR: Cannot reach {m}. Check API key and connectivity.")
            return

    # Load functions
    programs = load_benchmark(
        args.benchmark,
        min_complexity=args.min_complexity,
        max_functions=args.num_functions,
        shuffle=True,
        seed=42,
    )
    print(f"\n  Loaded {len(programs)} functions")

    # Run
    start = time.time()
    all_data = run_phase0b(programs, K=args.K, models=models)
    elapsed = time.time() - start
    cost = get_cost()

    # Analyze and report each estimator
    all_analyses = {}
    for est in ESTIMATOR_NAMES:
        analysis = analyze_results(all_data, est)
        all_analyses[est] = analysis
        print_report(est, analysis)

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Estimator':<35} {'rho':>8} {'p':>10} {'Verdict':>10}")
    print(f"  {'─' * 65}")
    for est in ESTIMATOR_NAMES:
        ov = all_analyses[est].get("overall", {})
        rho = ov.get("spearman_rho", 0)
        pval = ov.get("p_value", 1)
        if rho > 0.15:
            verdict = "GO"
        elif rho > 0.05:
            verdict = "PIVOT"
        else:
            verdict = "STOP"
        print(f"  {est:<35} {rho:>+8.4f} {pval:>10.4f} {verdict:>10}")

    # Decision table
    a_rho = all_analyses["multi_model_disagreement"]["overall"].get("spearman_rho", 0)
    c_rho = all_analyses["verbalized_confidence"]["overall"].get("spearman_rho", 0)
    d_rho = all_analyses["hybrid"]["overall"].get("spearman_rho", 0)

    best_est = max(ESTIMATOR_NAMES, key=lambda e: all_analyses[e]["overall"].get("spearman_rho", 0))
    best_rho = all_analyses[best_est]["overall"].get("spearman_rho", 0)

    print(f"\n  Best estimator: {best_est} (rho={best_rho:+.4f})")
    if best_rho > 0.15:
        print(f"  Recommendation: PROCEED to Phase 1b with {best_est}")
    elif best_rho > 0.05:
        print(f"  Recommendation: Investigate further — weak signal in {best_est}")
    else:
        print(f"  Recommendation: All estimators failed. Consider STOP or alternative approaches.")

    # Cost breakdown
    print(f"\n{'=' * 70}")
    print("COST BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"  Total API calls: {cost['api_calls']}")
    print(f"  Total tokens: {cost['total_tokens']:,}")
    print(f"  Total cost: ${cost['total_cost_usd']:.4f}")
    if cost.get("per_model"):
        for model, usage in cost["per_model"].items():
            print(f"    {model}: {usage['api_calls']} calls, "
                  f"{usage['input_tokens'] + usage['output_tokens']:,} tokens, "
                  f"${usage['cost_usd']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "phase0b_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "ensemble_models": models,
                "logprob_model": config.LOGPROB_MODEL,
                "verbalized_model": config.MODEL,
                "benchmark": args.benchmark,
                "num_functions": args.num_functions,
                "K": args.K,
                "min_complexity": args.min_complexity,
                "states": CALIBRATION_STATES,
            },
            "analyses": all_analyses,
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
            "data": all_data,
        }, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
