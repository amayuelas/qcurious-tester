#!/bin/bash
# Extra seeds for the "+ Function scorer" vs "+ Bayesian value" rows of the
# ablation table (seed 42 is in ablation_components_gemma.json). Waits for the
# ablation queue on :8001 to finish so the vLLM server is not oversubscribed,
# then runs both variants on seeds 123 and 456 with the same settings.
cd /share/edc/home/aamayuelasfernandez/qcurious-tester
export MODEL=gemma-4-31B-it DOCKER_MEMORY=2g DOCKER_CPUS=1
export VLLM_API_BASE=http://localhost:8001/v1
WAIT_PID=$1
STEM=ablation_fnt_vs_bayes_seeds_gemma

while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] ${STEM} ..."
.venv/bin/python run_repo_explore_bench.py --seeds 123 456 --parallel 48 \
    --strategies cov_qvalue_tgt_fnt cov_qvalue_tgt_fnt_bayes \
    --K 3 --exec-budget 24 --gamma 0.5 \
    --output "ablations/${STEM}.json" > "results/ablations/${STEM}.log" 2>&1
echo "[$(date +%H:%M)] ${STEM} exit=$?"
