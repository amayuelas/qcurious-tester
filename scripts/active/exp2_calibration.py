"""Experiment 2 — Q-value calibration study.

Reads a `cov_qvalue_calib` run (produced by run_repo_explore_bench.py with
--strategies cov_qvalue_calib), which logs for every round a list of K
candidate plans with the LLM-predicted scores (immediate ĝ, future v̂,
Q = ĝ + γ·v̂) AND each plan's REALIZED branch gain (measured by trial-running
all K from a snapshot). This gives paired (predicted, realized) data the
original cov_qvalue logs lack.

Reports:
  - Pooled Spearman/Pearson between predicted (Q, ĝ) and realized gain.
  - Within-round mean Spearman (rank-order is what selection actually uses).
  - Selection accuracy: how often the argmax-Q plan is (among) the best-realized
    of its K, vs the 1/K random baseline. Reported over all rounds and over
    "decisive" rounds (those with a unique realized best).
  - Regret: realized gain left on the table by the scorer's pick vs the oracle
    best-of-K (ties directly to Exp 1's oracle headroom).
  - A binned calibration curve (predicted Q bin -> mean realized gain).

Runs purely off the logged run — no API/Docker needed.

Usage:
    python scripts/active/exp2_calibration.py \
        --input results/repo_explore_bench/exp2_calib_gemini.json \
        --output results/repo_explore_bench/exp2_calibration.json
"""

import argparse
import json
import statistics
from pathlib import Path

from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_calib_rounds(path, strategy):
    """Yield each round's candidate list across all targets in the run."""
    data = json.load(open(path))
    rounds = []
    for r in data["results"]:
        strat = r.get("strategies", {}).get(strategy)
        if not strat or "calib" not in strat:
            continue
        for rd in strat["calib"]:
            cands = rd.get("candidates", [])
            if cands:
                rounds.append({
                    "module": r.get("module"),
                    "repo": r.get("repo"),
                    "selected_idx": rd.get("selected_idx"),
                    "candidates": cands,
                })
    return rounds


def pooled_corr(pairs):
    """Spearman + Pearson over a list of (predicted, realized) pairs."""
    if len(pairs) < 3:
        return None
    pred, real = zip(*pairs)
    if len(set(pred)) < 2 or len(set(real)) < 2:
        return None
    sr, sp = sp_stats.spearmanr(pred, real)
    pr, pp = sp_stats.pearsonr(pred, real)
    return {"spearman": sr, "spearman_p": sp,
            "pearson": pr, "pearson_p": pp, "n": len(pairs)}


def within_round_spearman(rounds, pred_key):
    """Mean per-round Spearman between predicted score and realized gain.

    Only rounds with >=2 candidates and variance in both axes contribute.
    """
    rhos = []
    for rd in rounds:
        cands = rd["candidates"]
        if len(cands) < 2:
            continue
        pred = [c[pred_key] for c in cands]
        real = [c["realized_gain"] for c in cands]
        if len(set(pred)) < 2 or len(set(real)) < 2:
            continue
        rho, _ = sp_stats.spearmanr(pred, real)
        if rho == rho:  # not nan
            rhos.append(rho)
    if not rhos:
        return None
    return {"mean_rho": statistics.mean(rhos),
            "se": statistics.stdev(rhos) / len(rhos) ** 0.5 if len(rhos) > 1 else 0,
            "n_rounds": len(rhos)}


def selection_accuracy(rounds):
    """Selection accuracy of argmax-Q vs realized best-of-K, plus regret."""
    n_all = 0
    hit_all = 0          # selected is among the realized-best (ties count as hit)
    n_decisive = 0       # rounds with a unique realized best
    hit_decisive = 0
    chance_sum = 0.0     # sum of 1/K -> expected random accuracy
    regrets = []         # oracle_best_realized - selected_realized
    for rd in rounds:
        cands = rd["candidates"]
        K = len(cands)
        if K < 2:
            continue
        sel = rd["selected_idx"]
        if sel is None or sel >= K:
            continue
        real = [c["realized_gain"] for c in cands]
        best = max(real)
        sel_real = real[sel]
        n_all += 1
        chance_sum += 1.0 / K
        if sel_real == best:
            hit_all += 1
        regrets.append(best - sel_real)
        # decisive: exactly one plan attains the max
        if real.count(best) == 1:
            n_decisive += 1
            if sel_real == best:
                hit_decisive += 1
    return {
        "n_rounds": n_all,
        "accuracy_all": hit_all / n_all if n_all else None,
        "chance_all": chance_sum / n_all if n_all else None,
        "n_decisive": n_decisive,
        "accuracy_decisive": hit_decisive / n_decisive if n_decisive else None,
        "mean_regret": statistics.mean(regrets) if regrets else None,
        "total_regret": sum(regrets) if regrets else None,
    }


def calibration_curve(pairs, n_bins=6):
    """Bin predicted Q, report mean realized gain per bin."""
    if not pairs:
        return []
    pred = [p for p, _ in pairs]
    lo, hi = min(pred), max(pred)
    if hi == lo:
        return []
    width = (hi - lo) / n_bins
    bins = [[] for _ in range(n_bins)]
    for p, r in pairs:
        idx = min(int((p - lo) / width), n_bins - 1)
        bins[idx].append(r)
    curve = []
    for i, b in enumerate(bins):
        if not b:
            continue
        curve.append({
            "q_lo": round(lo + i * width, 2),
            "q_hi": round(lo + (i + 1) * width, 2),
            "mean_realized": round(statistics.mean(b), 3),
            "n": len(b),
        })
    return curve


def main():
    ap = argparse.ArgumentParser(description="Exp 2 — Q-value calibration")
    ap.add_argument("--input",
                    default="repo_explore_bench/exp2_calib_gemini.json",
                    help="Path to the calib run JSON (relative to results/ "
                         "or absolute).")
    ap.add_argument("--strategy", default="cov_qvalue_calib")
    ap.add_argument("--output",
                    default="repo_explore_bench/exp2_calibration.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = RESULTS_DIR / in_path

    rounds = load_calib_rounds(in_path, args.strategy)
    if not rounds:
        print(f"No calib rounds found in {in_path} for strategy "
              f"'{args.strategy}'. Did the run use --strategies "
              f"{args.strategy}?")
        return

    q_pairs = [(c["predicted_q"], c["realized_gain"])
               for rd in rounds for c in rd["candidates"]]
    g_pairs = [(c["predicted_immediate"], c["realized_gain"])
               for rd in rounds for c in rd["candidates"]]

    report = {
        "input": str(in_path),
        "strategy": args.strategy,
        "n_rounds": len(rounds),
        "n_candidates": len(q_pairs),
        "pooled_q_vs_realized": pooled_corr(q_pairs),
        "pooled_immediate_vs_realized": pooled_corr(g_pairs),
        "within_round_q_spearman": within_round_spearman(rounds, "predicted_q"),
        "selection": selection_accuracy(rounds),
        "calibration_curve_q": calibration_curve(q_pairs),
    }

    # ---- Print summary ----
    print("=" * 64)
    print(f"Exp 2 — Q-value calibration  ({args.strategy})")
    print(f"  source: {in_path.name}")
    print(f"  rounds: {report['n_rounds']}  candidates: {report['n_candidates']}")
    print("=" * 64)

    pq = report["pooled_q_vs_realized"]
    if pq:
        print(f"\nPooled Q vs realized gain (n={pq['n']}):")
        print(f"  Spearman ρ={pq['spearman']:+.3f} (p={pq['spearman_p']:.3g})")
        print(f"  Pearson  r={pq['pearson']:+.3f} (p={pq['pearson_p']:.3g})")
    pg = report["pooled_immediate_vs_realized"]
    if pg:
        print(f"\nPooled ĝ (immediate) vs realized gain (n={pg['n']}):")
        print(f"  Spearman ρ={pg['spearman']:+.3f}  Pearson r={pg['pearson']:+.3f}")

    wr = report["within_round_q_spearman"]
    if wr:
        print(f"\nWithin-round Q rank corr (rank-order drives selection):")
        print(f"  mean ρ={wr['mean_rho']:+.3f} ± {wr['se']:.3f} "
              f"over {wr['n_rounds']} rounds with variance")

    s = report["selection"]
    print(f"\nSelection accuracy (argmax-Q picks the best-realized of K):")
    if s["accuracy_all"] is not None:
        print(f"  all rounds:      {s['accuracy_all']:.1%}  "
              f"(chance {s['chance_all']:.1%}, n={s['n_rounds']})")
    if s["accuracy_decisive"] is not None:
        print(f"  decisive rounds: {s['accuracy_decisive']:.1%}  "
              f"(unique best exists, n={s['n_decisive']})")
    if s["mean_regret"] is not None:
        print(f"  mean regret vs oracle best-of-K: {s['mean_regret']:.2f} "
              f"branches/round  (total {s['total_regret']:.0f})")

    cc = report["calibration_curve_q"]
    if cc:
        print(f"\nCalibration curve (predicted Q bin -> mean realized gain):")
        for b in cc:
            print(f"  Q∈[{b['q_lo']:>5}, {b['q_hi']:>5}]  "
                  f"realized={b['mean_realized']:>6}  (n={b['n']})")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = RESULTS_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
