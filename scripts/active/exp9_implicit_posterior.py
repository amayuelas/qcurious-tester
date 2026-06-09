"""Experiment 9 — Implicit posterior validation.

Operationalizes "the LLM acts as a world model whose posterior updates with
experience." Each round, the cov_bayes strategy elicits the LLM's prediction of
which functions a plan will exercise (q_f) — a sample from its implicit
p(O | h, a) — and we observe which functions the plan actually covered. This
script (reading the `exp9` logs in a cov_bayes run) tests two hypotheses:

  H1: prediction accuracy (precision/recall of predicted vs actually-covered
      functions) IMPROVES as the coverage map grows.
  H2: predictive entropy (mean binary entropy of the per-function confidences)
      DECREASES as exploration proceeds — posterior concentration.

Pure offline analysis of a cov_bayes run's logs — no API/Docker.

Usage:
    python scripts/active/exp9_implicit_posterior.py \
        --input repo_explore_bench/exp9_covbayes_gemma.json
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats as sp

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _H(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def load_rounds(path):
    data = json.load(open(path))
    rounds = []
    for r in data["results"]:
        for s in r.get("strategies", {}).values():
            for rd in s.get("exp9", []):
                rounds.append(rd)
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="repo_explore_bench/exp9_covbayes_gemma.json")
    ap.add_argument("--output", default="repo_explore_bench/exp9_implicit_posterior.json")
    args = ap.parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = RESULTS_DIR / path

    rounds = load_rounds(path)
    if not rounds:
        print(f"No exp9 logs in {path}. Run a cov_bayes strategy first.")
        return

    recs = []
    for rd in rounds:
        pred = rd.get("predicted", {})            # {func: q}
        actual = set(rd.get("newly_covered", []))
        if not pred:
            continue
        pset = set(pred)
        tp = len(pset & actual)
        precision = tp / len(pset) if pset else None
        recall = tp / len(actual) if actual else None
        mean_ent = statistics.mean(_H(q) for q in pred.values())
        recs.append({
            "n_covered_before": rd["n_covered_before"],
            "round": rd["round"],
            "precision": precision, "recall": recall,
            "mean_entropy": mean_ent, "n_pred": len(pred),
        })

    print("=" * 64)
    print(f"Exp 9 — implicit posterior validation  ({path.name})")
    print(f"  rounds with predictions: {len(recs)}")
    print("=" * 64)

    # H1: precision/recall vs coverage-map size (n_covered_before)
    pr = [(r["n_covered_before"], r["precision"]) for r in recs if r["precision"] is not None]
    rc = [(r["n_covered_before"], r["recall"]) for r in recs if r["recall"] is not None]
    out = {"n_rounds": len(recs)}
    print("\nH1 — prediction accuracy vs coverage-map size (Spearman ρ):")
    for label, pairs in [("precision", pr), ("recall", rc)]:
        if len(pairs) >= 5 and len({x for x, _ in pairs}) > 2:
            rho, p = sp.spearmanr([x for x, _ in pairs], [y for _, y in pairs])
            print(f"  {label:<10} vs map size: ρ={rho:+.3f} (p={p:.3g}, n={len(pairs)}); "
                  f"overall mean={statistics.mean([y for _, y in pairs]):.2f}")
            out[f"{label}_vs_mapsize_rho"] = rho
            out[f"{label}_vs_mapsize_p"] = p
            out[f"{label}_mean"] = statistics.mean([y for _, y in pairs])

    # binned view
    print("\n  binned by coverage-map size:")
    bins = [(0, 5), (5, 15), (15, 30), (30, 1e9)]
    for lo, hi in bins:
        b = [r for r in recs if lo <= r["n_covered_before"] < hi]
        if not b:
            continue
        pp = [r["precision"] for r in b if r["precision"] is not None]
        rr = [r["recall"] for r in b if r["recall"] is not None]
        print(f"    map∈[{lo:>2},{hi if hi<1e9 else '∞':>3}): n={len(b):<4} "
              f"precision={statistics.mean(pp):.2f} recall={statistics.mean(rr):.2f} "
              f"meanH={statistics.mean([r['mean_entropy'] for r in b]):.2f}")

    # H2: predictive entropy vs round index
    er = [(r["round"], r["mean_entropy"]) for r in recs]
    if len(er) >= 5:
        rho, p = sp.spearmanr([x for x, _ in er], [y for _, y in er])
        print(f"\nH2 — predictive entropy vs round index: ρ={rho:+.3f} (p={p:.3g})  "
              f"(negative = concentrates as exploration proceeds)")
        out["entropy_vs_round_rho"] = rho
        out["entropy_vs_round_p"] = p

    (RESULTS_DIR / args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {RESULTS_DIR / args.output}")


if __name__ == "__main__":
    main()
