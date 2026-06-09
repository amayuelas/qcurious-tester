#!/bin/bash
# Rebuttal pipeline — relaunch after server crash.
# Stage 1 (parallel): gemma K=8 (vLLM, parallel=8) + gemini K=3 (API, parallel=8) → ~16 concurrent dockers
# Stage 2 (serial):   gpt-5.4-mini → mistral (parallel=4 for 429s) → gpt-5.4 full
set -uo pipefail
cd /share/edc/home/aamayuelasfernandez/qcurious-tester

LOGDIR=results/repo_explore_bench
mkdir -p "$LOGDIR"
PIPELINE_LOG=/tmp/rebuttal_pipeline.log
exec >>"$PIPELINE_LOG" 2>&1

echo "==== $(date) START ===="

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

# Wait up to 30 min for vLLM
for i in $(seq 1 360); do
    if curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31B-it; then
        echo "[$(date)] vLLM ready after ~$((i*5))s"
        break
    fi
    sleep 5
done
if ! curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31B-it; then
    echo "[$(date)] ERROR: vLLM did not start within 30min. Aborting."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# --- Stage 1: gemma K=8 + gemini K=3 in parallel ---
echo "[$(date)] STAGE 1: gemma K=8 + gemini K=3 in parallel (parallel=8 each)"

( MODEL=gemma-4-31B-it python run_repo_explore_bench.py \
        --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_rank \
        --K 8 --seeds 42 --exec-budget 24 --parallel 8 \
        --output repo_explore_bench/exp_kr_k8_gemma.json \
        > "$LOGDIR/exp_kr_k8_gemma.log" 2>&1 ) &
GEMMA_PID=$!

( MODEL=gemini-3-flash-preview python run_repo_explore_bench.py \
        --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_rank \
        --K 3 --seeds 42 --exec-budget 24 --parallel 8 \
        --output repo_explore_bench/exp_mm_gemini.json \
        > "$LOGDIR/exp_mm_gemini.log" 2>&1 ) &
GEMINI_PID=$!

echo "[$(date)] gemma PID=$GEMMA_PID  gemini PID=$GEMINI_PID"
wait $GEMMA_PID; G1=$?
wait $GEMINI_PID; G2=$?
echo "[$(date)] STAGE 1 DONE (gemma exit=$G1, gemini exit=$G2)"

# Free GPUs — gemma vLLM no longer needed.
echo "[$(date)] stopping vLLM..."
kill $VLLM_PID 2>/dev/null
sleep 5
pkill -f "vllm serve google/gemma" 2>/dev/null
sleep 3

# --- Stage 2: gpt-5.4-mini ---
echo "[$(date)] STAGE 2a: gpt-5.4-mini K=3 (parallel=16)"
MODEL=gpt-5.4-mini python run_repo_explore_bench.py \
    --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
    --K 3 --seeds 42 --exec-budget 24 --parallel 16 \
    --output repo_explore_bench/exp_mm_gpt.json \
    > "$LOGDIR/exp_mm_gpt.log" 2>&1
echo "[$(date)] gpt-5.4-mini DONE (exit=$?)"

# --- Stage 2b: mistral (lower parallel for 429s) ---
echo "[$(date)] STAGE 2b: mistral-large-latest K=3 (parallel=4 — 429 throttling)"
MODEL=mistral-large-latest python run_repo_explore_bench.py \
    --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
    --K 3 --seeds 42 --exec-budget 24 --parallel 4 \
    --output repo_explore_bench/exp_mm_mistral.json \
    > "$LOGDIR/exp_mm_mistral.log" 2>&1
echo "[$(date)] mistral DONE (exit=$?)"

# --- Stage 2c: gpt-5.4 full ---
echo "[$(date)] STAGE 2c: gpt-5.4 full K=3 (parallel=16)"
MODEL=gpt-5.4 python run_repo_explore_bench.py \
    --strategies divhints_random divhints_oracle cov_qvalue cov_qvalue_calib \
    --K 3 --seeds 42 --exec-budget 24 --parallel 16 \
    --output repo_explore_bench/exp_mm_gpt54.json \
    > "$LOGDIR/exp_mm_gpt54.log" 2>&1
echo "[$(date)] gpt-5.4 full DONE (exit=$?)"

echo "==== $(date) PIPELINE COMPLETE ===="
