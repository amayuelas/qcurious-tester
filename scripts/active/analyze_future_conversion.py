"""Does the LLM's future term predict real future coverage?

For the plan committed at each round of a function-scorer calib run
(cov_qvalue*_fnv / _fnt), the scorer predicted two sets of still-uncovered
functions: EXECUTES (immediate) and ENABLES (unlocked for later). Using the
uncovered-function snapshot logged at every decision point, this measures,
for horizons h = 1, 2 rounds after the commit:

  conversion(ENABLES)  fraction of predicted-enabled functions (not covered
                       by the plan itself) that get covered within h rounds
  conversion(other)    the same for every other uncovered function (base rate)

Both by count and weighted by function size (lines). If ENABLES converts no
faster than the base rate, the future term carries no signal. The
size-weighted ENABLES conversion is an empirical discount: the share of the
predicted future value v that is realized within h rounds, i.e. a data-driven
estimate of gamma to set against the paper's gamma = 0.5.

Usage:
    python scripts/active/analyze_future_conversion.py \
        --input results/repo_explore_bench/pilot_fnt_gemma.json
"""

import argparse
import json


def _sizes(uncovered):
    return {q: max(e - s + 1, 1) for q, s, e in uncovered}


def conversion(rounds_by_target, h):
    acc = {"E": [0, 0, 0.0, 0.0], "B": [0, 0, 0.0, 0.0],  # hit, n, hit_mass, mass
           "self_E": 0, "n_E_total": 0}
    for rounds in rounds_by_target:
        for t, rd in enumerate(rounds):
            if t + 1 + h >= len(rounds) or not rd.get("uncovered"):
                continue
            sel = rd["candidates"][rd["selected_idx"]]
            E = set(sel.get("enables") or [])
            X = set(sel.get("executes") or [])
            size = _sizes(rd["uncovered"])
            U_next = {f[0] for f in rounds[t + 1]["uncovered"]}
            U_later = {f[0] for f in rounds[t + 1 + h]["uncovered"]}
            acc["n_E_total"] += len(E)
            acc["self_E"] += len(E - U_next)  # covered by the plan itself
            for group, fset in (("E", E & U_next),
                                ("B", U_next - E - X)):
                for f in fset:
                    hit = f not in U_later
                    a = acc[group]
                    a[0] += hit
                    a[1] += 1
                    a[2] += size.get(f, 1) * hit
                    a[3] += size.get(f, 1)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    data = json.load(open(args.input))
    strategies = [s for s in data["config"]["strategies"]
                  if "fnv" in s or "fnt" in s]

    for s in strategies:
        rounds_by_target = [r["strategies"][s].get("calib", [])
                            for r in data["results"] if s in r["strategies"]]
        print(f"\n{s}")
        for h in (1, 2):
            a = conversion(rounds_by_target, h)
            E, B = a["E"], a["B"]
            if not E[1] or not B[1]:
                print(f"  h={h}: not enough data")
                continue
            print(f"  h={h}: ENABLES converts {E[0]/E[1]:.1%} of functions "
                  f"({E[2]/E[3]:.1%} of lines, n={E[1]})  vs other uncovered "
                  f"{B[0]/B[1]:.1%} ({B[2]/B[3]:.1%} of lines, n={B[1]})  "
                  f"-> lift {(E[0]/E[1])/(B[0]/B[1]) if B[0] else float('inf'):.1f}x")
        a = conversion(rounds_by_target, 1)
        if a["n_E_total"]:
            print(f"  predicted-enabled functions already covered by the plan "
                  f"itself: {a['self_E']/a['n_E_total']:.0%}")


if __name__ == "__main__":
    main()
