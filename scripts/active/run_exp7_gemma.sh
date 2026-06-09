#!/bin/bash
# Exp 7 on gemma-4-31B (headline model, K=8) — second-model replication of the
# summary-content ablation for Reviewer c8fs. Free (local vLLM); requires the
# gemma server already up at localhost:8000 (started separately).
#
# 4 modes, all same-harness paired runs (cov_qvalue_calib gives coverage AND
# top-1 selection accuracy). Sequential — one vLLM server, and the runner's
# parallel=8 already saturates it.
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/exp7_gemma_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START Exp7 gemma K=8 ===="

# Guard: server must be reachable before we start.
if ! curl -s -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31B-it; then
    echo "[$(date)] ERROR: gemma vLLM not reachable at :8000 — aborting."; exit 1
fi

for MODE in full stats exemplars none; do
    echo "[$(date)] STAGE: map-mode=$MODE"
    MODEL=gemma-4-31B-it .venv/bin/python run_repo_explore_bench.py \
        --strategies cov_qvalue_calib \
        --map-mode "$MODE" \
        --K 8 --seeds 42 --exec-budget 24 --parallel 8 \
        --output "repo_explore_bench/exp7_${MODE}_gemma.json" \
        > "$LOGDIR/exp7_${MODE}_gemma.log" 2>&1
    echo "[$(date)] DONE map-mode=$MODE (exit=$?)"
done

echo "==== $(date) Exp7 gemma COMPLETE ===="
