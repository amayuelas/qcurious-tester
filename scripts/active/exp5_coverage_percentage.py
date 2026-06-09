"""Experiment 5 — Coverage-percentage metric.

Addresses sHcP's concern that absolute branch counts are outlier-dominated.
Recomputes per-strategy coverage as a PERCENTAGE of the per-module total, so
high-branch modules no longer dominate the mean.

Denominator (per module): the maximum final branch count achieved by ANY
strategy across ALL models. This is a lower bound on the truly reachable
branch set and is the best estimate available from existing logs, which store
branch *counts* rather than branch *identifiers* (so the exact cross-strategy
union is not recoverable from logs — see note below).

NOTE: For a true union denominator the runner must serialize branch IDs per
step. That instrumentation is being added for the new rebuttal runs; until
then "max-achieved" is the defensible proxy and conclusions should be checked
for robustness against the absolute-count tables.

Runs purely off existing result logs — no API/Docker needed.

Usage:
    python scripts/active/exp5_coverage_percentage.py
    python scripts/active/exp5_coverage_percentage.py --benchmark repo_explore_bench
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS = ["gemini", "gpt54mini", "mistral"]


def load_runs(benchmark):
    runs = {}
    for model in MODELS:
        path = RESULTS_DIR / benchmark / f"full_run_{model}.json"
        if path.exists():
            runs[model] = json.load(open(path))["results"]
    # Fixed-scorer reruns (cov_qvalue no longer truncated; gemma adds a whole
    # new model with all 7 strategies incl. the oracle ceiling, which makes the
    # per-module denominator less self-referential to cov_qvalue).
    for label, fname in [("gemini_fixed", "exp1fixed_gemini.json"),
                         ("gemma_4_31b", "exp1fixed_gemma.json")]:
        path = RESULTS_DIR / benchmark / fname
        if path.exists():
            runs[label] = json.load(open(path))["results"]
    return runs


def build_denominators(runs):
    """Per module: max final branch count across all strategies × all models."""
    denom = {}
    for results in runs.values():
        for r in results:
            mod = r["module"]
            for sdata in r["strategies"].values():
                denom[mod] = max(denom.get(mod, 0), sdata.get("final", 0))
    return denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="repo_explore_bench")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    runs = load_runs(args.benchmark)
    if not runs:
        print(f"No result files for '{args.benchmark}' in {RESULTS_DIR}")
        return

    denom = build_denominators(runs)
    usable = {m: d for m, d in denom.items() if d > 0}
    print(f"Loaded models: {list(runs.keys())}  | modules with denom>0: "
          f"{len(usable)}/{len(denom)}\n")

    strategies = sorted({s for results in runs.values() for r in results
                         for s in r["strategies"]})

    out = {"benchmark": args.benchmark, "denominator": "max-achieved per module",
           "per_model": {}, "robustness": {}}

    # Per-model: mean coverage % across modules, alongside mean absolute count
    for model, results in runs.items():
        print(f"=== {model} ===")
        print(f"  {'strategy':<14} {'mean %':>8} {'± SE':>7} {'mean abs':>9}")
        per_strat_pct = {}
        for strat in strategies:
            pcts, absol = [], []
            for r in results:
                if strat not in r["strategies"]:
                    continue
                mod = r["module"]
                d = denom.get(mod, 0)
                if d <= 0:
                    continue
                fin = r["strategies"][strat]["final"]
                pcts.append(100.0 * fin / d)
                absol.append(fin)
            if not pcts:
                continue
            mean_pct = statistics.mean(pcts)
            se = statistics.stdev(pcts) / len(pcts) ** 0.5 if len(pcts) > 1 else 0
            per_strat_pct[strat] = pcts
            print(f"  {strat:<14} {mean_pct:>7.1f}% {se:>6.2f} "
                  f"{statistics.mean(absol):>9.1f}")
            out["per_model"].setdefault(model, {})[strat] = {
                "mean_pct": mean_pct, "se": se,
                "mean_abs": statistics.mean(absol), "n": len(pcts)}

        # Robustness: does the % ranking match the absolute ranking?
        rank_pct = sorted(per_strat_pct, key=lambda s: -statistics.mean(per_strat_pct[s]))
        print(f"  ranking by %: {rank_pct}")
        out["robustness"][model] = {"ranking_by_pct": rank_pct}
        print()

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
