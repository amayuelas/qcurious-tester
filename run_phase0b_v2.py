"""Phase 0b v2: Improved estimators based on trace analysis.

Three improved estimators:
  A' — Logprob entropy + per-function z-score normalization
  B' — Contrastive ranking (show all K candidates, ask to rank)
  C' — P(yes) from logprobs (single call, continuous probability)

Uses the same functions/states as Phase 0b v1 for direct comparison.

Usage:
    python run_phase0b_v2.py --num-functions 5 --K 5    # smoke test
    python run_phase0b_v2.py                             # full run
"""

import argparse
import json
import logging
import math
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ESTIMATOR_NAMES = ["logprob_entropy_zscore", "contrastive_rank", "p_yes"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0b v2: Improved estimators"
    )
    parser.add_argument("--num-functions", type=int, default=30)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--min-complexity", type=int, default=20)
    parser.add_argument("--benchmark", default="ult", choices=["ult", "cruxeval"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Estimator A': Logprob entropy (raw — z-scoring done in post-processing)
# ---------------------------------------------------------------------------

def compute_logprob_entropy(func_name, source_code, test_history,
                            candidate_test):
    """Single-call logprob entropy via Fireworks model."""
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history)
    prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    result = generate_with_logprobs(config.LOGPROB_MODEL, prompt,
                                    temperature=0.3, max_tokens=100,
                                    top_logprobs=5)
    if not result or not result["token_logprobs"]:
        return 0.0

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

    return sum(entropies) / len(entropies) if entropies else 0.0


# ---------------------------------------------------------------------------
# Estimator B': Contrastive ranking
# ---------------------------------------------------------------------------

def compute_contrastive_ranking(func_name, source_code, test_history,
                                candidates):
    """Show all K candidates at once, ask model to rank them.

    Returns dict mapping candidate_test -> rank_score (higher = better).
    """
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history)
    # Build numbered candidate list
    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""Given this function:
{code_section}

{history_str}

Here are {len(candidates)} candidate test calls. Rank them from MOST to LEAST likely to discover NEW, previously-unseen behavior (new branches, new code paths, new edge cases).

Candidates:
{cand_list}

Respond with ONLY the ranking as comma-separated numbers (best first).
Example for 5 candidates: 3,1,5,2,4"""

    response = generate_with_model(config.MODEL, prompt, temperature=0.3,
                                   max_tokens=100)

    # Parse ranking
    ranks = _parse_ranking(response, len(candidates))

    # Convert ranks to scores: rank 1 → highest score
    n = len(candidates)
    scores = {}
    for i, cand in enumerate(candidates):
        rank = ranks.get(i + 1, n)  # default to worst if not ranked
        scores[cand] = (n - rank + 1) / n  # normalize to [0, 1]

    return scores


def _parse_ranking(response, n_candidates):
    """Parse a ranking response like '3,1,5,2,4' into {position: rank}."""
    # Extract numbers from response
    numbers = re.findall(r'\d+', response)
    ranks = {}
    seen = set()
    rank = 1
    for num_str in numbers:
        num = int(num_str)
        if 1 <= num <= n_candidates and num not in seen:
            ranks[num] = rank
            seen.add(num)
            rank += 1
        if rank > n_candidates:
            break

    # Fill in any missing candidates with worst rank
    for i in range(1, n_candidates + 1):
        if i not in ranks:
            ranks[i] = rank
            rank += 1

    return ranks


# ---------------------------------------------------------------------------
# Estimator C': P(yes) from logprobs
# ---------------------------------------------------------------------------

def compute_p_yes(func_name, source_code, test_history, candidate_test,
                  cumulative_arcs=0):
    """Ask model if test will find new branches, extract P(yes) from logprobs.

    For reasoning models (like gpt-oss-120b), the model thinks before answering.
    We use enough tokens to get past reasoning, then extract P(yes) from the
    FIRST token in the reasoning chain where yes/no appears as a top-k
    alternative — this captures the model's pre-commitment assessment.

    Inspired by Kadavath et al. P(True) approach.
    """
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history)
    if test_history:
        recent = test_history[-5:]
        history_str = "Previous tests and results:\n"
        for test_code, result in recent:
            history_str += (f"  {test_code} → "
                           f"new_branches={result.new_branches}, "
                           f"output={result.output or result.exception}\n")

    prompt = f"""Given this function:
{code_section}

{history_str}
Branches covered so far: {cumulative_arcs}

Will the following test discover NEW branches not yet covered by previous tests?
Test: {candidate_test}

Answer with ONLY "yes" or "no"."""

    result = generate_with_logprobs(config.LOGPROB_MODEL, prompt,
                                    temperature=0.0, max_tokens=200,
                                    top_logprobs=5)
    if not result or not result["token_logprobs"]:
        return 0.5

    # Strategy: collect ALL tokens where yes/no appears in top-k.
    # The FIRST such token is the model's initial assessment (most informative).
    # The LAST such token is the final answer (often deterministic, less useful).
    # We use the first occurrence where both yes and no are plausible.
    yes_no_tokens = []

    for i, tok in enumerate(result["token_logprobs"]):
        top_lps = tok.get("top_logprobs", {})

        p_yes_lp = None
        p_no_lp = None
        for t, lp in top_lps.items():
            t_clean = t.strip().lower().rstrip(".,!?")
            if t_clean == "yes":
                p_yes_lp = lp
            elif t_clean in ("no", "none"):
                p_no_lp = lp

        # Also check the token itself
        tok_clean = tok["token"].strip().lower().rstrip(".,!?")
        if tok_clean == "yes" and p_yes_lp is None:
            p_yes_lp = tok["logprob"]
        elif tok_clean in ("no", "none") and p_no_lp is None:
            p_no_lp = tok["logprob"]

        if p_yes_lp is not None or p_no_lp is not None:
            yes_no_tokens.append({
                "idx": i,
                "token": tok["token"],
                "p_yes_lp": p_yes_lp,
                "p_no_lp": p_no_lp,
            })

    if not yes_no_tokens:
        # Fallback: parse final text
        text = result["text"].strip().lower()
        if text.startswith("yes") or text.endswith("yes"):
            return 0.7
        elif text.startswith("no") or text.endswith("no"):
            return 0.3
        return 0.5

    # Find the first token where BOTH yes and no have meaningful probability.
    # This is the real decision point in the reasoning chain.
    for yt in yes_no_tokens:
        if yt["p_yes_lp"] is not None and yt["p_no_lp"] is not None:
            p_y = math.exp(yt["p_yes_lp"])
            p_n = math.exp(yt["p_no_lp"])
            total = p_y + p_n
            if total > 0.01:  # both have non-trivial probability
                return p_y / total

    # If no joint token found, use the first yes/no token we saw
    yt = yes_no_tokens[0]
    if yt["p_yes_lp"] is not None and yt["p_no_lp"] is not None:
        p_y = math.exp(yt["p_yes_lp"])
        p_n = math.exp(yt["p_no_lp"])
        return p_y / (p_y + p_n) if (p_y + p_n) > 0 else 0.5
    elif yt["p_yes_lp"] is not None:
        return math.exp(yt["p_yes_lp"])
    else:
        return 1.0 - math.exp(yt["p_no_lp"])


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def build_up_state(func_name, source, runner, test_history, target_count, K=3):
    """Run tests to build exploration state."""
    while len(test_history) < target_count:
        candidates = generate_test_candidates(
            func_name, source, test_history=test_history, K=K
        )
        if not candidates:
            break
        result = runner.run_test(candidates[0])
        test_history.append((candidates[0], result))


def score_candidates_v2(func_name, source, test_history, runner, candidates):
    """Score all candidates with the three improved estimators."""
    results = []

    # --- A': Logprob entropy for each candidate (parallel) ---
    lpe_scores = {}
    with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
        futures = {
            executor.submit(compute_logprob_entropy, func_name, source,
                            test_history, c): c
            for c in candidates
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                lpe_scores[c] = future.result()
            except Exception as e:
                log.warning(f"LPE failed: {e}")
                lpe_scores[c] = 0.0

    # --- B': Contrastive ranking (single call for all candidates) ---
    rank_scores = compute_contrastive_ranking(
        func_name, source, test_history, candidates
    )

    # --- C': P(yes) for each candidate (parallel) ---
    cumulative_arcs = runner.get_cumulative_coverage()
    pyes_scores = {}
    with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
        futures = {
            executor.submit(compute_p_yes, func_name, source, test_history,
                            c, cumulative_arcs): c
            for c in candidates
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                pyes_scores[c] = future.result()
            except Exception as e:
                log.warning(f"P(yes) failed: {e}")
                pyes_scores[c] = 0.5

    # --- Actual coverage (ground truth) ---
    for c in candidates:
        temp_runner = CoverageRunner(func_name, source)
        temp_runner.cumulative_branches = set(runner.cumulative_branches)
        result = temp_runner.run_test(c)

        results.append({
            "test": c,
            "logprob_entropy_raw": round(lpe_scores.get(c, 0), 4),
            "contrastive_rank": round(rank_scores.get(c, 0.5), 4),
            "p_yes": round(pyes_scores.get(c, 0.5), 4),
            "new_branches": result.new_branches,
            "output": result.output,
            "exception": result.exception,
        })

    # --- A' post-processing: z-score within this batch ---
    lpe_vals = [r["logprob_entropy_raw"] for r in results]
    if len(lpe_vals) > 1 and statistics.stdev(lpe_vals) > 0.001:
        mu = statistics.mean(lpe_vals)
        sigma = statistics.stdev(lpe_vals)
        for r in results:
            r["logprob_entropy_zscore"] = round(
                (r["logprob_entropy_raw"] - mu) / sigma, 4
            )
    else:
        for r in results:
            r["logprob_entropy_zscore"] = 0.0

    return results


def run_experiment(programs, K):
    """Run the v2 experiment."""
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

            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K
            )
            if not candidates:
                print("    No candidates, skipping")
                continue

            points = score_candidates_v2(
                func_name, source, test_history, runner, candidates
            )

            for pt in points:
                pt["state"] = state
                pt["complexity"] = complexity
                pt["func_key"] = key
                pt["func_name"] = func_name
                all_data.append(pt)

            # Per-state summary
            gains = [p["new_branches"] for p in points]
            for est in ESTIMATOR_NAMES:
                scores = [p.get(est, 0) for p in points]
                if len(points) >= 3 and len(set(scores)) > 1 and len(set(gains)) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rho, _ = stats.spearmanr(scores, gains)
                    print(f"    [{est[:14]:<14}] rho={rho:+.3f} | "
                          f"mean={statistics.mean(scores):.3f} | "
                          f"gain={statistics.mean(gains):.1f}")

    return all_data


def analyze_results(all_data, estimator_name):
    """Compute correlations overall, by state, and within function×state."""
    results = {}
    scores = [d.get(estimator_name, 0) for d in all_data]
    gains = [d["new_branches"] for d in all_data]

    # Overall
    if len(all_data) >= 3 and len(set(scores)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, pval = stats.spearmanr(scores, gains)
        results["overall"] = {
            "n": len(all_data),
            "spearman_rho": round(rho, 4) if rho == rho else 0,
            "p_value": round(pval, 6) if pval == pval else 1,
        }
    else:
        results["overall"] = {"n": len(all_data), "spearman_rho": 0, "p_value": 1}

    # By state
    results["by_state"] = {}
    for state in CALIBRATION_STATES:
        subset = [d for d in all_data if d["state"] == state]
        s = [d.get(estimator_name, 0) for d in subset]
        g = [d["new_branches"] for d in subset]
        if len(subset) >= 3 and len(set(s)) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, pval = stats.spearmanr(s, g)
            results["by_state"][f"state_{state}"] = {
                "n": len(subset),
                "spearman_rho": round(rho, 4) if rho == rho else 0,
                "p_value": round(pval, 6) if pval == pval else 1,
            }

    # Within function×state (the real test)
    within_rhos = []
    combos = set((d["func_key"], d["state"]) for d in all_data)
    for fk, st in combos:
        subset = [d for d in all_data if d["func_key"] == fk and d["state"] == st]
        s = [d.get(estimator_name, 0) for d in subset]
        g = [d["new_branches"] for d in subset]
        if len(subset) >= 3 and len(set(s)) > 1 and len(set(g)) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = stats.spearmanr(s, g)
            if rho == rho:
                within_rhos.append(rho)

    results["within_func_state"] = {
        "n_groups": len(within_rhos),
        "mean_rho": round(statistics.mean(within_rhos), 4) if within_rhos else 0,
        "n_positive": sum(1 for r in within_rhos if r > 0),
        "rhos": [round(r, 3) for r in within_rhos],
    }

    return results


def print_report(est_name, analysis):
    labels = {
        "logprob_entropy_zscore": "Logprob Entropy + Z-Score (A')",
        "contrastive_rank": "Contrastive Ranking (B')",
        "p_yes": "P(yes) from Logprobs (C')",
    }
    label = labels.get(est_name, est_name)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    ov = analysis["overall"]
    rho = ov["spearman_rho"]
    print(f"\n  Overall: n={ov['n']}, rho={rho:+.4f}, p={ov['p_value']}")

    if analysis["by_state"]:
        print("\n  By state:")
        for label, data in analysis["by_state"].items():
            print(f"    [{label}] n={data['n']}, rho={data['spearman_rho']:+.4f} "
                  f"(p={data['p_value']:.4f})")

    wfs = analysis["within_func_state"]
    print(f"\n  Within function×state (THE KEY METRIC):")
    print(f"    Groups with variance: {wfs['n_groups']}")
    print(f"    Mean within-group rho: {wfs['mean_rho']:+.4f}")
    print(f"    Positive: {wfs['n_positive']}/{wfs['n_groups']}")
    if wfs["rhos"]:
        print(f"    Individual rhos: {wfs['rhos']}")

    # Verdict based on within-group metric
    print(f"\n  {'─' * 50}")
    if wfs["n_groups"] >= 2 and wfs["mean_rho"] > 0.15:
        print(f"  VERDICT: GO (within-group rho={wfs['mean_rho']:+.3f} > 0.15)")
    elif wfs["n_groups"] >= 2 and wfs["mean_rho"] > 0.05:
        print(f"  VERDICT: PIVOT (within-group rho={wfs['mean_rho']:+.3f})")
    elif wfs["n_groups"] < 2:
        print(f"  VERDICT: INSUFFICIENT DATA ({wfs['n_groups']} groups)")
    else:
        print(f"  VERDICT: STOP (within-group rho={wfs['mean_rho']:+.3f})")


def main():
    args = parse_args()
    reset_cost()

    print("=" * 70)
    print("Phase 0b v2: Improved Estimators")
    print("=" * 70)
    print(f"  Logprob model: {config.LOGPROB_MODEL}")
    print(f"  Ranking model: {config.MODEL}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Functions: {args.num_functions} (complexity >= {args.min_complexity})")
    print(f"  K={args.K} candidates per state")
    print(f"  States: {CALIBRATION_STATES}")
    print(f"  Estimators: {ESTIMATOR_NAMES}")
    print("=" * 70)

    # Connectivity check
    print("\nTesting model connectivity...")
    for m in [config.MODEL, config.LOGPROB_MODEL]:
        test = generate_with_model(m, "Say 'ok'", temperature=0.3, max_tokens=10)
        status = "OK" if test else "FAILED"
        print(f"  {m}: {status}" + (f" ({test[:20]})" if test else ""))
        if not test:
            print(f"\n  ERROR: Cannot reach {m}")
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
    all_data = run_experiment(programs, K=args.K)
    elapsed = time.time() - start
    cost = get_cost()

    # Analyze
    all_analyses = {}
    for est in ESTIMATOR_NAMES:
        analysis = analyze_results(all_data, est)
        all_analyses[est] = analysis
        print_report(est, analysis)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — Within func×state discrimination (the metric that matters)")
    print(f"{'=' * 70}")
    print(f"  {'Estimator':<30} {'Mean ρ':>8} {'Positive':>10} {'Verdict':>10}")
    print(f"  {'─' * 60}")
    for est in ESTIMATOR_NAMES:
        wfs = all_analyses[est]["within_func_state"]
        mr = wfs["mean_rho"]
        pos = f"{wfs['n_positive']}/{wfs['n_groups']}"
        if wfs["n_groups"] >= 2 and mr > 0.15:
            verdict = "GO"
        elif wfs["n_groups"] >= 2 and mr > 0.05:
            verdict = "PIVOT"
        elif wfs["n_groups"] < 2:
            verdict = "INSUFF"
        else:
            verdict = "STOP"
        print(f"  {est:<30} {mr:>+8.4f} {pos:>10} {verdict:>10}")

    best = max(ESTIMATOR_NAMES,
               key=lambda e: all_analyses[e]["within_func_state"]["mean_rho"])
    best_rho = all_analyses[best]["within_func_state"]["mean_rho"]
    print(f"\n  Best: {best} (within-group ρ={best_rho:+.4f})")

    # Cost
    print(f"\n{'=' * 70}")
    print("COST")
    print(f"{'=' * 70}")
    print(f"  API calls: {cost['api_calls']}")
    print(f"  Tokens: {cost['total_tokens']:,}")
    print(f"  Cost: ${cost['total_cost_usd']:.4f}")
    if cost.get("per_model"):
        for model, usage in cost["per_model"].items():
            print(f"    {model}: {usage['api_calls']} calls, ${usage['cost_usd']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "phase0b_v2_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "logprob_model": config.LOGPROB_MODEL,
                "ranking_model": config.MODEL,
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
