"""Compare CovQValue variants from one calib run: _fb (scorer feedback) and
_tgt (uncovered-function-targeted generation).

Every strategy is a cov_qvalue[_fb][_tgt]_calib run, so each round logs all K
candidates' predicted Q and realized gain (trial-run from a snapshot). That
separates the two proposed changes cleanly:

  - generation quality: mean realized gain of the candidate pool (what a
    random pick from the pool would get) and the best-of-K (oracle)
  - selection quality: top-1 accuracy vs chance, and the share of the
    random->oracle gap the argmax-Q pick closes, pooled over rounds
  - end to end: final branches, paired against the baseline variant

Usage:
    python scripts/active/analyze_pilot_fbtgt.py \
        --input results/repo_explore_bench/pilot_fbtgt_gemma.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2_calibration import load_calib_rounds, selection_accuracy  # noqa: E402

BASELINE = "cov_qvalue_calib"


def pool_stats(rounds):
    """Pool mean / oracle / selected realized gain, summed over rounds."""
    pool = oracle = sel = 0.0
    for rd in rounds:
        real = [c["realized_gain"] for c in rd["candidates"]]
        pool += statistics.mean(real)
        oracle += max(real)
        sel += real[rd["selected_idx"]]
    gap = oracle - pool
    n = len(rounds) or 1
    return {
        "rounds": len(rounds),
        "pool_mean_per_round": pool / n,
        "oracle_per_round": oracle / n,
        "selected_per_round": sel / n,
        "oracle_gap_closed": (sel - pool) / gap if gap else None,
    }


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    w = sum(x > 0 for x in d)
    l = sum(x < 0 for x in d)
    p = sp_stats.wilcoxon(a, b).pvalue if any(d) else 1.0
    se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0
    return statistics.mean(d), se, w, l, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--baseline", default=BASELINE,
                    help="strategy the others are paired against")
    args = ap.parse_args()
    baseline = args.baseline

    data = json.load(open(args.input))
    strategies = data["config"]["strategies"]
    results = [r for r in data["results"]
               if all(s in r["strategies"] for s in strategies)]
    print(f"{args.input}: {len(results)} targets with all strategies\n")

    finals = {s: [r["strategies"][s]["final"] for r in results]
              for s in strategies}

    print(f"{'strategy':<26}{'final':>14}{'Δ vs base (p)':>22}{'W/L':>8}")
    for s in strategies:
        v = finals[s]
        se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0
        row = f"{s:<26}{statistics.mean(v):>8.1f} ± {se:<4.1f}"
        if s != baseline and baseline in finals:
            m, dse, w, l, p = paired(v, finals[baseline])
            row += f"{m:>+10.1f} ± {dse:<4.1f}({p:.3f}){w:>5}/{l}"
        print(row)

    print(f"\n{'strategy':<26}{'pool':>7}{'oracle':>8}{'picked':>8}"
          f"{'gap closed':>12}{'top1 (decisive)':>18}{'chance':>8}")
    for s in strategies:
        rounds = load_calib_rounds(args.input, s)
        rounds = [r for r in rounds if len(r["candidates"]) >= 2]
        ps = pool_stats(rounds)
        acc = selection_accuracy(rounds)
        gc = ps["oracle_gap_closed"]
        print(f"{s:<26}{ps['pool_mean_per_round']:>7.2f}"
              f"{ps['oracle_per_round']:>8.2f}{ps['selected_per_round']:>8.2f}"
              f"{(f'{gc:.0%}' if gc is not None else '-'):>12}"
              f"{acc['accuracy_decisive'] or 0:>11.1%} (n={acc['n_decisive']})"
              f"{acc['chance_all'] or 0:>8.1%}")


if __name__ == "__main__":
    main()
