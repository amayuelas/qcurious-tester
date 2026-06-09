#!/bin/bash
# Exp 7 — summary-content ablation (Reviewer c8fs follow-up).
# Isolates whether the actionable signal in the coverage-state summary is the
# aggregate statistics or the exemplar pointers. Runs cov_qvalue_calib (gives
# BOTH end-to-end coverage AND per-round top-1 selection accuracy) under three
# summary modes; "full" is the already-completed exp_calib_gemini.json baseline.
#
# gemini K=3, 93 targets, budget 24, seed 42. Sequential (not parallel) because
# llm.py has no 429 backoff — concurrent gemini bursts would silently fail
# scoring calls (empty -> 0) and bias the selection-accuracy measurement.
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/exp7_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START Exp7 summary-content ablation ===="

for MODE in stats exemplars none; do
    echo "[$(date)] STAGE: map-mode=$MODE"
    MODEL=gemini-3-flash-preview .venv/bin/python run_repo_explore_bench.py \
        --strategies cov_qvalue_calib \
        --map-mode "$MODE" \
        --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
        --output "repo_explore_bench/exp7_${MODE}_gemini.json" \
        > "$LOGDIR/exp7_${MODE}_gemini.log" 2>&1
    echo "[$(date)] DONE map-mode=$MODE (exit=$?)"
done

echo "==== $(date) Exp7 COMPLETE ===="
