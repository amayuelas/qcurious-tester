"""Phase 0b v3: Measure estimator correlation with BOTH targets.

Target 1: Immediate coverage gain (new_branches) — greedy metric
Target 2: Learning progress — did this test teach the model something?
  - Prediction surprise: was model's prediction wrong?
  - Next-step quality: are the next candidates better after this test?

Uses logprob entropy (A') as the primary estimator + contrastive ranking (B').

Usage:
    python run_phase0b_v3.py --num-functions 5 --K 5    # smoke test
    python run_phase0b_v3.py                             # full run
"""

import argparse
import json
import logging
import math
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from scipy import stats

import config
from curiosity_explorer.llm import (
    generate_with_model, generate_with_logprobs, get_cost, reset_cost,
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

CALIBRATION_STATES = [0, 5, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0b v3: Two-target calibration")
    parser.add_argument("--num-functions", type=int, default=30)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--min-complexity", type=int, default=20)
    parser.add_argument("--benchmark", default="ult", choices=["ult", "cruxeval"])
    parser.add_argument("--next-step-K", type=int, default=5,
                        help="Candidates to generate for next-step quality (default: 5)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logprob entropy + predicted output (returns both score and prediction)
# ---------------------------------------------------------------------------

def compute_logprob_entropy_with_prediction(func_name, source_code, test_history,
                                            candidate_test):
    """Returns (entropy_score, predicted_output_text)."""
    code_section, history_str = _build_context(func_name, source_code, test_history)

    prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    result = generate_with_logprobs(config.LOGPROB_MODEL, prompt,
                                    temperature=0.3, max_tokens=100,
                                    top_logprobs=5)
    if not result or not result["token_logprobs"]:
        return 0.0, ""

    # Compute mean per-token entropy
    entropies = []
    for tok in result["token_logprobs"]:
        top_lps = tok.get("top_logprobs", {})
        if not top_lps:
            continue
        probs = [math.exp(lp) for lp in top_lps.values()]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        entropies.append(ent)

    entropy = sum(entropies) / len(entropies) if entropies else 0.0
    predicted_output = result["text"].strip()

    return entropy, predicted_output


# ---------------------------------------------------------------------------
# Prediction surprise
# ---------------------------------------------------------------------------

def compute_surprise(predicted: str, actual: str) -> dict:
    """Compare predicted vs actual output. Returns surprise metrics."""
    pred_clean = predicted.strip().lower()[:200]
    actual_clean = actual.strip().lower()[:200]

    exact_match = pred_clean == actual_clean
    # Fuzzy similarity (0 = completely different, 1 = identical)
    similarity = SequenceMatcher(None, pred_clean, actual_clean).ratio()
    # Surprise = 1 - similarity (high when prediction was wrong)
    surprise = 1.0 - similarity

    return {
        "exact_match": exact_match,
        "similarity": round(similarity, 4),
        "surprise": round(surprise, 4),
    }


# ---------------------------------------------------------------------------
# Next-step quality: after incorporating test X, how good are the next candidates?
# ---------------------------------------------------------------------------

def measure_next_step_quality(func_name, source, runner, test_history,
                              candidate_test, candidate_result, next_K=5):
    """Execute candidate, add to history, generate next candidates, measure their quality.

    Returns: max coverage gain among next-step candidates.
    """
    # Create updated history with this candidate's result
    updated_history = list(test_history) + [(candidate_test, candidate_result)]

    # Generate next-step candidates
    next_candidates = generate_test_candidates(
        func_name, source, test_history=updated_history, K=next_K
    )
    if not next_candidates:
        return 0

    # Measure coverage of each next-step candidate
    # (from the state AFTER executing the current candidate)
    updated_branches = set(runner.cumulative_branches)
    # Add branches from the current candidate
    temp = CoverageRunner(func_name, source)
    temp.cumulative_branches = set(runner.cumulative_branches)
    temp_result = temp.run_test(candidate_test)
    updated_branches = temp.cumulative_branches

    gains = []
    for nc in next_candidates:
        nc_runner = CoverageRunner(func_name, source)
        nc_runner.cumulative_branches = set(updated_branches)
        nc_result = nc_runner.run_test(nc)
        gains.append(nc_result.new_branches)

    return max(gains) if gains else 0


# ---------------------------------------------------------------------------
# Score all candidates for a function×state
# ---------------------------------------------------------------------------

def score_candidates(func_name, source, test_history, runner, candidates, next_K):
    """Score candidates with estimator + measure both targets."""
    results = []

    # --- Logprob entropy + prediction for each candidate (parallel) ---
    lpe_data = {}
    with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
        futures = {
            executor.submit(compute_logprob_entropy_with_prediction,
                            func_name, source, test_history, c): c
            for c in candidates
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                entropy, prediction = future.result()
                lpe_data[c] = {"entropy": entropy, "prediction": prediction}
            except Exception as e:
                log.warning(f"LPE failed: {e}")
                lpe_data[c] = {"entropy": 0.0, "prediction": ""}

    # --- Execute each candidate + compute surprise ---
    for c in candidates:
        temp_runner = CoverageRunner(func_name, source)
        temp_runner.cumulative_branches = set(runner.cumulative_branches)
        result = temp_runner.run_test(c)

        actual_output = result.output or result.exception or ""
        predicted = lpe_data.get(c, {}).get("prediction", "")
        surprise_data = compute_surprise(predicted, actual_output)

        results.append({
            "test": c,
            "logprob_entropy_raw": round(lpe_data.get(c, {}).get("entropy", 0), 4),
            "predicted_output": predicted[:100],
            "actual_output": actual_output[:100],
            # Target 1: Immediate gain
            "new_branches": result.new_branches,
            # Target 2a: Prediction surprise
            "surprise": surprise_data["surprise"],
            "exact_match": surprise_data["exact_match"],
            "similarity": surprise_data["similarity"],
            # Result object for next-step measurement
            "_result": result,
        })

    # --- Z-score entropy within batch ---
    ent_vals = [r["logprob_entropy_raw"] for r in results]
    if len(ent_vals) > 1 and statistics.stdev(ent_vals) > 0.001:
        mu = statistics.mean(ent_vals)
        sigma = statistics.stdev(ent_vals)
        for r in results:
            r["logprob_entropy_zscore"] = round((r["logprob_entropy_raw"] - mu) / sigma, 4)
    else:
        for r in results:
            r["logprob_entropy_zscore"] = 0.0

    # --- Next-step quality for top and bottom scoring candidates ---
    # Sort by entropy (raw, since z-score preserves rank)
    sorted_by_score = sorted(results, key=lambda r: r["logprob_entropy_raw"], reverse=True)
    top_cand = sorted_by_score[0]
    bottom_cand = sorted_by_score[-1]

    for r in results:
        if r is top_cand or r is bottom_cand:
            nsq = measure_next_step_quality(
                func_name, source, runner, test_history,
                r["test"], r["_result"], next_K=next_K,
            )
            r["next_step_quality"] = nsq
        else:
            r["next_step_quality"] = None

    # Clean up non-serializable data
    for r in results:
        del r["_result"]

    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def build_up_state(func_name, source, runner, test_history, target_count, K=3):
    while len(test_history) < target_count:
        candidates = generate_test_candidates(
            func_name, source, test_history=test_history, K=K
        )
        if not candidates:
            break
        result = runner.run_test(candidates[0])
        test_history.append((candidates[0], result))


def run_experiment(programs, K, next_K):
    all_data = []
    total = len(programs)

    for i, (key, prog) in enumerate(programs.items()):
        func_name = prog["func_name"]
        source = prog["source"]
        complexity = prog.get("metadata", {}).get("cyclomatic_complexity", 0)

        print(f"\n[{i+1}/{total}] {key} ({func_name}, complexity={complexity})",
              flush=True)

        runner = CoverageRunner(func_name, source)
        test_history = []

        for state in CALIBRATION_STATES:
            build_up_state(func_name, source, runner, test_history, state, K=3)

            print(f"  State {state} (arcs={runner.get_cumulative_coverage()}): "
                  f"scoring {K} candidates...", flush=True)

            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K
            )
            if not candidates:
                print("    No candidates, skipping", flush=True)
                continue

            points = score_candidates(
                func_name, source, test_history, runner, candidates, next_K
            )

            for pt in points:
                pt["state"] = state
                pt["complexity"] = complexity
                pt["func_key"] = key
                pt["func_name"] = func_name
                all_data.append(pt)

            # Print summary
            gains = [p["new_branches"] for p in points]
            surprises = [p["surprise"] for p in points]
            entropies = [p["logprob_entropy_zscore"] for p in points]

            # Next-step comparison
            nsq_top = [p["next_step_quality"] for p in points if p["next_step_quality"] is not None]

            if len(set(entropies)) > 1 and len(set(gains)) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho_gain, _ = stats.spearmanr(entropies, gains)
                print(f"    entropy vs gain:     rho={rho_gain:+.3f}", flush=True)

            if len(set(entropies)) > 1 and len(set(surprises)) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho_surp, _ = stats.spearmanr(entropies, surprises)
                print(f"    entropy vs surprise: rho={rho_surp:+.3f}", flush=True)

            if len(nsq_top) == 2:
                top_nsq, bottom_nsq = nsq_top[0], nsq_top[1]
                # top = highest entropy candidate, bottom = lowest
                print(f"    next-step quality: high_ent={top_nsq}, low_ent={bottom_nsq}",
                      flush=True)

    return all_data


def analyze(all_data, score_key, target_key):
    """Compute within-group Spearman for score_key vs target_key."""
    within_rhos = []
    combos = set((d["func_key"], d["state"]) for d in all_data)
    for fk, st in combos:
        subset = [d for d in all_data if d["func_key"] == fk and d["state"] == st]
        s = [d.get(score_key, 0) for d in subset]
        t = [d.get(target_key, 0) for d in subset]
        if len(subset) >= 3 and len(set(s)) > 1 and len(set(t)) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = stats.spearmanr(s, t)
            if rho == rho:
                within_rhos.append(rho)

    if not within_rhos:
        return {"n_groups": 0, "mean_rho": 0, "n_positive": 0}
    return {
        "n_groups": len(within_rhos),
        "mean_rho": round(statistics.mean(within_rhos), 4),
        "n_positive": sum(1 for r in within_rhos if r > 0),
        "rhos": [round(r, 3) for r in within_rhos],
    }


def main():
    args = parse_args()
    reset_cost()

    print("=" * 70, flush=True)
    print("Phase 0b v3: Two-Target Calibration", flush=True)
    print("=" * 70, flush=True)
    print(f"  Logprob model: {config.LOGPROB_MODEL}", flush=True)
    print(f"  Candidate gen model: {config.MODEL}", flush=True)
    print(f"  Functions: {args.num_functions} (complexity >= {args.min_complexity})", flush=True)
    print(f"  K={args.K}, next-step K={args.next_step_K}", flush=True)
    print(f"  States: {CALIBRATION_STATES}", flush=True)
    print("=" * 70, flush=True)

    # Connectivity
    print("\nConnectivity check...", flush=True)
    for m in [config.MODEL, config.LOGPROB_MODEL]:
        test = generate_with_model(m, "Say 'ok'", temperature=0.3, max_tokens=10)
        print(f"  {m}: {'OK' if test else 'FAILED'}", flush=True)
        if not test:
            return

    # Load
    programs = load_benchmark(
        args.benchmark, min_complexity=args.min_complexity,
        max_functions=args.num_functions, shuffle=True, seed=42,
    )
    print(f"  Loaded {len(programs)} functions\n", flush=True)

    start = time.time()
    all_data = run_experiment(programs, K=args.K, next_K=args.next_step_K)
    elapsed = time.time() - start
    cost = get_cost()

    # === ANALYSIS ===
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS: Logprob Entropy vs Two Targets", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Target 1: immediate gain
    r1 = analyze(all_data, "logprob_entropy_zscore", "new_branches")
    print(f"\n  Target 1 — Immediate Coverage Gain (new_branches):", flush=True)
    print(f"    Within-group ρ: {r1['mean_rho']:+.4f} "
          f"({r1['n_positive']}/{r1['n_groups']} positive)", flush=True)
    if r1['mean_rho'] > 0.15:
        print(f"    Verdict: GO", flush=True)
    elif r1['mean_rho'] > 0.05:
        print(f"    Verdict: PIVOT", flush=True)
    else:
        print(f"    Verdict: STOP", flush=True)

    # Target 2a: prediction surprise
    r2 = analyze(all_data, "logprob_entropy_zscore", "surprise")
    print(f"\n  Target 2a — Prediction Surprise:", flush=True)
    print(f"    Within-group ρ: {r2['mean_rho']:+.4f} "
          f"({r2['n_positive']}/{r2['n_groups']} positive)", flush=True)
    if r2['mean_rho'] > 0.15:
        print(f"    Verdict: GO", flush=True)
    elif r2['mean_rho'] > 0.05:
        print(f"    Verdict: PIVOT", flush=True)
    else:
        print(f"    Verdict: STOP", flush=True)

    # Target 2b: next-step quality comparison
    print(f"\n  Target 2b — Next-Step Quality (high-entropy vs low-entropy pick):", flush=True)
    high_wins = 0
    low_wins = 0
    ties = 0
    comparisons = 0
    combos = set((d["func_key"], d["state"]) for d in all_data)
    for fk, st in combos:
        subset = [d for d in all_data if d["func_key"] == fk and d["state"] == st]
        with_nsq = [d for d in subset if d["next_step_quality"] is not None]
        if len(with_nsq) == 2:
            # First is top (highest entropy), second is bottom (lowest entropy)
            sorted_by_ent = sorted(with_nsq, key=lambda d: d["logprob_entropy_raw"],
                                   reverse=True)
            high_nsq = sorted_by_ent[0]["next_step_quality"]
            low_nsq = sorted_by_ent[1]["next_step_quality"]
            comparisons += 1
            if high_nsq > low_nsq:
                high_wins += 1
            elif low_nsq > high_nsq:
                low_wins += 1
            else:
                ties += 1

    print(f"    Comparisons: {comparisons}", flush=True)
    print(f"    High-entropy pick leads to better next step: {high_wins}/{comparisons}", flush=True)
    print(f"    Low-entropy pick leads to better next step: {low_wins}/{comparisons}", flush=True)
    print(f"    Ties: {ties}/{comparisons}", flush=True)
    if comparisons > 0:
        win_rate = high_wins / comparisons
        print(f"    Win rate: {win_rate:.1%} (50% = random)", flush=True)

    # Surprise distribution
    all_surprises = [d["surprise"] for d in all_data]
    all_matches = [d["exact_match"] for d in all_data]
    print(f"\n  Prediction accuracy: {sum(all_matches)}/{len(all_matches)} exact matches "
          f"({100*sum(all_matches)/len(all_matches):.0f}%)", flush=True)
    print(f"  Mean surprise: {statistics.mean(all_surprises):.3f}", flush=True)

    # Cost
    print(f"\n{'=' * 70}", flush=True)
    print(f"  Cost: ${cost['total_cost_usd']:.4f} | "
          f"Calls: {cost['api_calls']} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "phase0b_v3_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "logprob_model": config.LOGPROB_MODEL,
                "candidate_model": config.MODEL,
                "benchmark": args.benchmark,
                "num_functions": args.num_functions,
                "K": args.K,
                "next_step_K": args.next_step_K,
                "min_complexity": args.min_complexity,
            },
            "results": {
                "entropy_vs_gain": r1,
                "entropy_vs_surprise": r2,
                "next_step": {
                    "comparisons": comparisons,
                    "high_entropy_wins": high_wins,
                    "low_entropy_wins": low_wins,
                    "ties": ties,
                },
            },
            "cost": cost,
            "elapsed_seconds": round(elapsed, 1),
            "data": all_data,
        }, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
