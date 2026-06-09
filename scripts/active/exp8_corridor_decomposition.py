"""Experiment 8 — Per-step gain decomposition (corridor evidence).

Classifies every executed test/step in the logged runs as:
  - surface : high immediate branch gain (> median)
  - corridor: low immediate gain (<= 25th pct) FOLLOWED BY a downstream jump
              (branches gained in the next `horizon` steps >= jump threshold)
  - bust    : low immediate gain (<= 25th pct) with no downstream jump

Reports the fraction of each type per strategy and per model. Hypothesis:
CovQValue produces more corridor steps than greedy variants and fewer busts
than random.

Thresholds (median, 25th pct of immediate gain; jump threshold) are computed
once over the POOLED distribution of all steps within a benchmark so the
per-strategy fractions are directly comparable.

Runs purely off existing result logs — no API/Docker needed.

Usage:
    python scripts/active/exp8_corridor_decomposition.py
    python scripts/active/exp8_corridor_decomposition.py --benchmark repo_explore_bench
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS = ["gemini", "gpt54mini", "mistral"]
HORIZON = 3  # "next 3 rounds" downstream window


def load_runs(benchmark):
    """Load {model: results-list} for the given benchmark, skipping missing."""
    runs = {}
    for model in MODELS:
        path = RESULTS_DIR / benchmark / f"full_run_{model}.json"
        if path.exists():
            runs[model] = json.load(open(path))["results"]
    # Fixed-scorer reruns: these add divhints_random / divhints_oracle /
    # cov_qvalue_calib, so the decomposition can localize corridor-step behavior
    # to the generation pipeline vs the Q-value selection.
    for label, fname in [("gemini_fixed", "exp1fixed_gemini.json"),
                         ("gemma_4_31b", "exp1fixed_gemma.json")]:
        path = RESULTS_DIR / benchmark / fname
        if path.exists():
            runs[label] = json.load(open(path))["results"]
    return runs


def step_gains(trace, horizon):
    """Yield (immediate_gain, downstream_gain) for each step in a trace.

    downstream_gain = sum of new_branches over the next `horizon` steps.
    """
    imm = [t.get("new_branches", 0) for t in trace]
    for i in range(len(imm)):
        downstream = sum(imm[i + 1: i + 1 + horizon])
        yield imm[i], downstream


def collect(runs, horizon):
    """Return {(model, strategy): [(imm, downstream), ...]} and pooled imm list."""
    by_key = {}
    pooled_imm = []
    for model, results in runs.items():
        for r in results:
            for strat, sdata in r["strategies"].items():
                trace = sdata.get("trace", [])
                gains = list(step_gains(trace, horizon))
                by_key.setdefault((model, strat), []).extend(gains)
                pooled_imm.extend(g[0] for g in gains)
    return by_key, pooled_imm


def classify(imm, downstream, imm_median, imm_25, jump_thresh):
    if imm > imm_median:
        return "surface"
    if imm <= imm_25:
        return "corridor" if downstream >= jump_thresh else "bust"
    return "middle"  # low-but-not-bottom-quartile immediate gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="repo_explore_bench")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--output", default=None,
                    help="Optional JSON path to save the decomposition.")
    args = ap.parse_args()

    runs = load_runs(args.benchmark)
    if not runs:
        print(f"No result files found for benchmark '{args.benchmark}' in {RESULTS_DIR}")
        return
    print(f"Loaded models: {list(runs.keys())}  (benchmark={args.benchmark}, "
          f"horizon={args.horizon})")

    by_key, pooled_imm = collect(runs, args.horizon)

    # Pooled thresholds on immediate gain
    imm_median = statistics.median(pooled_imm)
    imm_sorted = sorted(pooled_imm)
    imm_25 = imm_sorted[max(0, int(0.25 * len(imm_sorted)) - 1)]
    # Jump threshold: a downstream window counts as a "jump" if it reveals at
    # least a median immediate-gain worth of branches (a real downstream payoff).
    jump_thresh = max(1, imm_median)
    print(f"Thresholds (pooled, n={len(pooled_imm)} steps): "
          f"imm_median={imm_median}, imm_25={imm_25}, jump>={jump_thresh}\n")

    strategies = sorted({k[1] for k in by_key})
    out = {"benchmark": args.benchmark, "horizon": args.horizon,
           "thresholds": {"imm_median": imm_median, "imm_25": imm_25,
                          "jump": jump_thresh}, "per_model": {}, "aggregate": {}}

    # Per-model tables
    for model in runs:
        print(f"=== {model} ===")
        print(f"  {'strategy':<16} {'n':>6} {'surface':>9} {'corridor':>9} "
              f"{'bust':>7} {'middle':>7}")
        for strat in strategies:
            gains = by_key.get((model, strat), [])
            if not gains:
                continue
            counts = {"surface": 0, "corridor": 0, "bust": 0, "middle": 0}
            for imm, ds in gains:
                counts[classify(imm, ds, imm_median, imm_25, jump_thresh)] += 1
            n = len(gains)
            frac = {k: v / n for k, v in counts.items()}
            print(f"  {strat:<16} {n:>6} {frac['surface']:>8.1%} "
                  f"{frac['corridor']:>8.1%} {frac['bust']:>6.1%} "
                  f"{frac['middle']:>6.1%}")
            out["per_model"].setdefault(model, {})[strat] = {
                "n": n, "counts": counts, "fractions": frac}
        print()

    # Threshold-free view: among zero-immediate-gain ("setup") steps, how often
    # is there a downstream payoff within the horizon? This avoids any
    # dependence on the (zero-inflated) median/percentile thresholds.
    print("=== THRESHOLD-FREE: setup-step payoff (aggregate over models) ===")
    print(f"  {'strategy':<16} {'n':>6} {'zero-gain':>10} "
          f"{'payoff|zero':>12} {'mean ds|zero':>13}")
    for strat in strategies:
        gains = [g for (m, s), gs in by_key.items() if s == strat for g in gs]
        if not gains:
            continue
        zero = [(imm, ds) for imm, ds in gains if imm == 0]
        n = len(gains)
        zero_rate = len(zero) / n
        payoff = (sum(1 for _, ds in zero if ds >= jump_thresh) / len(zero)
                  if zero else 0.0)
        mean_ds = (statistics.mean([ds for _, ds in zero]) if zero else 0.0)
        print(f"  {strat:<16} {n:>6} {zero_rate:>9.1%} "
              f"{payoff:>11.1%} {mean_ds:>13.2f}")
        out["aggregate"].setdefault(strat, {})
        out["aggregate"][strat]["setup_step"] = {
            "zero_gain_rate": zero_rate,
            "payoff_given_zero": payoff,
            "mean_downstream_given_zero": mean_ds,
        }
    print()

    # Aggregate across models
    print("=== AGGREGATE (all models) ===")
    print(f"  {'strategy':<16} {'n':>6} {'surface':>9} {'corridor':>9} "
          f"{'bust':>7} {'middle':>7}")
    for strat in strategies:
        gains = [g for (m, s), gs in by_key.items() if s == strat for g in gs]
        if not gains:
            continue
        counts = {"surface": 0, "corridor": 0, "bust": 0, "middle": 0}
        for imm, ds in gains:
            counts[classify(imm, ds, imm_median, imm_25, jump_thresh)] += 1
        n = len(gains)
        frac = {k: v / n for k, v in counts.items()}
        print(f"  {strat:<16} {n:>6} {frac['surface']:>8.1%} "
              f"{frac['corridor']:>8.1%} {frac['bust']:>6.1%} "
              f"{frac['middle']:>6.1%}")
        out["aggregate"][strat] = {"n": n, "counts": counts, "fractions": frac}

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
