#!/bin/bash
# Exp 7b — scorer-isolation refinement (Reviewer c8fs follow-up).
# Holds GENERATION fixed at full and ablates ONLY the Q-value scorer's summary.
# Directly tests whether the aggregate statistics / exemplars carry the SELECTION
# signal, separate from their generation role — the within-round-constant test.
# Baseline (scorer=full) is the existing exp_calib_gemini.json (top-1 ~80.5%).
#
# gemini K=3, 93 targets, budget 24, seed 42. Sequential (no 429 backoff in llm.py).
# Run AFTER run_exp7_summary_ablation.sh finishes to avoid concurrent rate limits.
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/exp7_scorer_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START Exp7b scorer-isolation ===="

for SMODE in stats exemplars none; do
    echo "[$(date)] STAGE: gen=full scorer=$SMODE"
    MODEL=gemini-3-flash-preview .venv/bin/python run_repo_explore_bench.py \
        --strategies cov_qvalue_calib \
        --map-mode full --score-map-mode "$SMODE" \
        --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
        --output "repo_explore_bench/exp7_scorer_${SMODE}_gemini.json" \
        > "$LOGDIR/exp7_scorer_${SMODE}_gemini.log" 2>&1
    echo "[$(date)] DONE scorer=$SMODE (exit=$?)"
done

echo "==== $(date) Exp7b COMPLETE ===="
