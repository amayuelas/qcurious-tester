"""Accuracy of the LLM's EXECUTES predictions (appendix table).

Each untested function at each scored round is one yes/no prediction: did the
scorer say the committed plan executes it, and did the plan actually execute
it (untested before the round, no longer untested after it)?

Reads either format:
  - RepoExploreBench calibration runs (strategy *_calib, field "calib"): the
    untested set after round t is the one logged at round t+1.
  - TestGenEval runs with prediction logging (field "pred_log").

Usage:
    python scripts/active/prediction_accuracy.py results/ablations/ablation_calibration_gemma.json
    python scripts/active/prediction_accuracy.py results/testgeneval/full_run_gemma4_predlog.json
"""

import json
import sys


def rounds_from(strategy_data):
    """Yield (untested_before, untested_after, predicted_executes)."""
    if "pred_log" in strategy_data:
        for r in strategy_data["pred_log"]:
            yield set(r["uncovered_before"]), set(r["uncovered_after"]), set(r["executes"])
    elif "calib" in strategy_data:
        rounds = strategy_data["calib"]
        for t in range(len(rounds) - 1):
            before = {u[0] for u in rounds[t]["uncovered"]}
            after = {u[0] for u in rounds[t + 1]["uncovered"]}
            sel = rounds[t]["candidates"][rounds[t]["selected_idx"]]
            yield before, after, set(sel.get("executes") or ())


def main(path):
    tp = fp = fn = tn = 0
    for r in json.load(open(path))["results"]:
        for s in r["strategies"].values():
            for before, after, pred in rounds_from(s):
                if not before:
                    continue
                executed = before - after
                pred &= before
                tp += len(pred & executed)
                fp += len(pred - executed)
                fn += len(executed - pred)
                tn += len(before - pred - executed)
    n = tp + fp + fn + tn
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    print(f"predictions {n}  accuracy {(tp + tn) / n:.2f}  precision {prec:.2f}  "
          f"recall {rec:.2f}  F1 {2 * prec * rec / (prec + rec):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
