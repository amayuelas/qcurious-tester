"""Experiment 3 (+ Exp 1) — component decomposition ladder.

Joins the Exp-3 run (cov_greedy, cov_greedy_multistep) with the Exp-1 run
(divhints_random, cov_qvalue, divhints_oracle) on module name to attribute the
cov_greedy -> cov_qvalue gap to each added component. All runs share config
(seed 42, budget 24, K=3) so per-module pairing is valid.

Ladder (each step adds exactly one component):
    cov_greedy            single-step, target-uncovered, random select
    + multistep  -------> cov_greedy_multistep
    + diversity  -------> divhints_random
    + Q-value    -------> cov_qvalue
    (ceiling)            divhints_oracle

Usage:
    python scripts/active/exp3_decomposition.py
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "repo_explore_bench"


def load(path, strat):
    """Return {module: final} for one strategy from a results file, or {}."""
    p = RESULTS / path
    if not p.exists():
        return {}
    d = json.load(open(p))
    out = {}
    for r in d["results"]:
        if strat in r["strategies"]:
            out[r["module"]] = r["strategies"][strat]["final"]
    return out


def paired(a_map, b_map):
    mods = sorted(set(a_map) & set(b_map))
    deltas = [a_map[m] - b_map[m] for m in mods]
    if len(deltas) < 2 or statistics.pstdev(deltas) == 0:
        return None
    from scipy import stats as sp
    sd = statistics.stdev(deltas)
    md = statistics.mean(deltas)
    t, p = sp.ttest_1samp(deltas, 0)
    return {"n": len(deltas), "mean_delta": md, "se": sd / len(deltas) ** 0.5,
            "wins": sum(1 for x in deltas if x > 0),
            "losses": sum(1 for x in deltas if x < 0),
            "ties": sum(1 for x in deltas if x == 0),
            "p": p, "cohens_d": md / sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3", default="exp3_gemini.json")
    ap.add_argument("--exp1", default="exp1_gemini.json")
    args = ap.parse_args()

    cg = load(args.exp3, "cov_greedy")
    cgm = load(args.exp3, "cov_greedy_multistep")
    dr = load(args.exp1, "divhints_random")
    cq = load(args.exp1, "cov_qvalue")
    orc = load(args.exp1, "divhints_oracle")

    rungs = [("cov_greedy", cg), ("cov_greedy_multistep", cgm),
             ("divhints_random", dr), ("cov_qvalue", cq),
             ("divhints_oracle", orc)]
    print("Per-strategy mean branches (modules present):")
    for name, m in rungs:
        if m:
            print(f"  {name:22} {statistics.mean(m.values()):6.1f}  (n={len(m)})")
    print()

    print("Component increments (paired):")
    for label, a, b in [
        ("+ multistep  (cgm - cg)", cgm, cg),
        ("+ diversity  (dr - cgm)", dr, cgm),
        ("+ Q-value    (cq - dr)", cq, dr),
        ("full gap     (cq - cg)", cq, cg),
        ("oracle headroom (orc - cq)", orc, cq),
    ]:
        r = paired(a, b)
        if r is None:
            print(f"  {label:30} — (missing data)")
            continue
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01
              else ("*" if r["p"] < 0.05 else ""))
        print(f"  {label:30} Δ={r['mean_delta']:+6.2f} ± {r['se']:.2f}  "
              f"W{r['wins']}/L{r['losses']}/T{r['ties']}  "
              f"p={r['p']:.4f} d={r['cohens_d']:+.2f} {sig}  (n={r['n']})")


if __name__ == "__main__":
    main()
