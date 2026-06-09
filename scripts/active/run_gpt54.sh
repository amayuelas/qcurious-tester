#!/bin/bash
# Lone gpt-5.4 full run at parallel=8 (halved from 16 after the mistral-induced crash).
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/gpt54_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START gpt-5.4 full ===="
MODEL=gpt-5.4 python run_repo_explore_bench.py \
    --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
    --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
    --output repo_explore_bench/exp_mm_gpt54.json \
    > "$LOGDIR/exp_mm_gpt54.log" 2>&1
echo "[$(date)] gpt-5.4 full DONE (exit=$?)"
echo "==== $(date) COMPLETE ===="
