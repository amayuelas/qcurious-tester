#!/bin/bash
# Gemma 4 CovQValue on TestGenEval Lite with prediction logging (pred_log), for
# the accuracy-of-predictions appendix. Uses a separate key (gemma4_predlog) so
# the main Gemma TestGenEval results are not touched. Waits for the extra-seed
# job using the same vLLM server (PID passed as $1; pass 0 to start now) so it is not
# oversubscribed. --keep-images: the GLM TestGenEval run uses the same images.
cd /share/edc/home/aamayuelasfernandez/qcurious-tester
export VLLM_API_BASE=http://localhost:8000/v1 DOCKER_MEMORY=2g DOCKER_CPUS=1
WAIT_PID=$1

while [ "$WAIT_PID" != 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] gemma4_predlog TestGenEval ..."
python scripts/active/run_tge_by_repo.py \
    --models gemma4_predlog=gemma-4-31B-it --strategies covqvalue2 \
    --exec-budget 24 --K 3 --gamma 0.5 --parallel 8 --keep-images
echo "[$(date +%H:%M)] gemma4_predlog exit=$?"
