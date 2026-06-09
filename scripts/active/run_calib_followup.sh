#!/bin/bash
# Phase 8 follow-up: within-pool calibration for the two significant models
# (gemma K=8, gemini K=3). cov_qvalue_calib swapped in for cov_qvalue_rank.
# The gemini K=3 run also serves as a reproduction of exp1fixed_gemini (same
# strategy set + seed) to settle whether the +1.2 vs +4.98 inconsistency is
# run-to-run noise or a real config difference.
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
PIPELINE_LOG=/tmp/calib_followup_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START calib follow-up ===="

# --- vLLM (gemma) ---
echo "[$(date)] starting vLLM..."
HF_HOME=/share/edc/home/aamayuelasfernandez/HUGGINGFACE \
nohup /local/home/aamayuelasfernandez/venvs/vllm-gemma/bin/vllm serve google/gemma-4-31B-it \
    --enable-auto-tool-choice --tool-call-parser gemma4 --host 0.0.0.0 \
    --max-model-len 8192 --served-model-name gemma-4-31B-it \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 0}' --max-num-batched-tokens 8192 \
    > /tmp/vllm_gemma.log 2>&1 &
VLLM_PID=$!
echo "[$(date)] vLLM PID=$VLLM_PID"

for i in $(seq 1 360); do
    if curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31B-it; then
        echo "[$(date)] vLLM ready after ~$((i*5))s"; break
    fi
    sleep 5
done
if ! curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31B-it; then
    echo "[$(date)] ERROR: vLLM failed to start"; kill $VLLM_PID 2>/dev/null; exit 1
fi

# --- Parallel pair: gemma K=8 calib + gemini K=3 calib ---
echo "[$(date)] STAGE: gemma K=8 + gemini K=3 (both with cov_qvalue_calib)"

( MODEL=gemma-4-31B-it python run_repo_explore_bench.py \
        --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
        --K 8 --seeds 42 --exec-budget 24 --parallel 8 \
        --output repo_explore_bench/exp_calib_k8_gemma.json \
        > "$LOGDIR/exp_calib_k8_gemma.log" 2>&1 ) &
GEMMA_PID=$!

( MODEL=gemini-3-flash-preview python run_repo_explore_bench.py \
        --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
        --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
        --output repo_explore_bench/exp_calib_gemini.json \
        > "$LOGDIR/exp_calib_gemini.log" 2>&1 ) &
GEMINI_PID=$!

echo "[$(date)] gemma PID=$GEMMA_PID  gemini PID=$GEMINI_PID"
wait $GEMMA_PID; G1=$?
wait $GEMINI_PID; G2=$?
echo "[$(date)] STAGE DONE (gemma exit=$G1, gemini exit=$G2)"

kill $VLLM_PID 2>/dev/null
sleep 5
pkill -f "vllm serve google/gemma" 2>/dev/null

echo "==== $(date) CALIB FOLLOWUP COMPLETE ===="
