#!/bin/bash
# Gemini K=3 calib alone (no GPU needed). Picks up half of the original
# calib followup; gemma K=8 calib waits for GPUs to free up.
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/calib_gemini_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START gemini K=3 calib ===="
MODEL=gemini-3-flash-preview python run_repo_explore_bench.py \
    --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
    --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
    --output repo_explore_bench/exp_calib_gemini.json \
    > "$LOGDIR/exp_calib_gemini.log" 2>&1
echo "[$(date)] gemini K=3 calib DONE (exit=$?)"
echo "==== $(date) COMPLETE ===="
