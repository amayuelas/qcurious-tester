"""Exp 7 — summary-content ablation analysis (Reviewer c8fs follow-up).

For each summary mode, report:
  (1) end-to-end branch coverage (mean final, from cov_qvalue_calib which commits
      the same argmax-Q plan as cov_qvalue), and
  (2) top-1 selection accuracy — how often the highest-Q plan was also the
      highest-realized-coverage plan, over rounds with >=2 distinct candidates.

This is the exact accuracy definition used for the full-mode 80.3% figure in the
reply. "full" is read from the existing exp_calib_gemini.json baseline; the other
modes from the exp7_<mode>_gemini.json runs.
"""
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "repo_explore_bench"

# mode -> (json file, strategy key)
SOURCES = {
    "full":      ("exp7_full_gemini.json", "cov_qvalue_calib"),
    "stats":     ("exp7_stats_gemini.json", "cov_qvalue_calib"),
    "exemplars": ("exp7_exemplars_gemini.json", "cov_qvalue_calib"),
    "none":      ("exp7_none_gemini.json", "cov_qvalue_calib"),
}


def analyze(path, strat):
    d = json.load(open(path))
    res = d["results"]
    finals, rounds, top1 = [], 0, 0
    for t in res:
        s = t["strategies"].get(strat)
        if not s:
            continue
        finals.append(s.get("final", 0))
        for r in s.get("calib", []):
            cands = r["candidates"]
            if len(cands) < 2:
                continue
            sel = r["selected_idx"]
            best = max(cands, key=lambda c: c["realized_gain"])["realized_gain"]
            if cands[sel]["realized_gain"] == best:
                top1 += 1
            rounds += 1
    return {
        "n_targets": len(finals),
        "mean_coverage": statistics.mean(finals) if finals else 0.0,
        "rounds": rounds,
        "top1_acc": (100 * top1 / rounds) if rounds else float("nan"),
    }


def main():
    print(f"{'mode':<11} {'cov(mean)':>10} {'top-1 acc':>10} {'rounds':>8} {'n':>5}")
    print("-" * 48)
    for mode, (fname, strat) in SOURCES.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"{mode:<11} {'(pending)':>10}   {fname}")
            continue
        a = analyze(path, strat)
        print(f"{mode:<11} {a['mean_coverage']:>10.2f} "
              f"{a['top1_acc']:>9.1f}% {a['rounds']:>8} {a['n_targets']:>5}")
    print("\nChance baseline (K=3): 33.3%")


if __name__ == "__main__":
    main()
