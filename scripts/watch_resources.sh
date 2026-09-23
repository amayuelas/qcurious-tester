#!/bin/bash
# Emit a line only when a resource threshold is crossed, so it can drive a
# monitor during high-parallelism runs. Silent while everything is healthy.
#
#   bash scripts/watch_resources.sh [interval_seconds]
#
# Warns on: low free RAM, container OOM kills, swap use, vLLM server down,
# and Docker daemon slowness (container start latency).
set -uo pipefail
INTERVAL=${1:-60}
MIN_FREE_GB=${MIN_FREE_GB:-40}
MAX_SWAP_GB=${MAX_SWAP_GB:-8}   # ~3GB is this host's baseline from other workloads

count_oom() { dmesg 2>/dev/null | grep -ci "killed process" | head -1 | tr -dc '0-9'; }
last_oom=$(count_oom); last_oom=${last_oom:-0}
while true; do
    free_gb=$(free -g | awk 'NR==2{print $7}')
    swap_gb=$(free -g | awk 'NR==3{print $3}')
    (( free_gb < MIN_FREE_GB )) && echo "WARN low RAM: ${free_gb}GB available (< ${MIN_FREE_GB})"
    (( swap_gb > MAX_SWAP_GB )) && echo "WARN swapping: ${swap_gb}GB swap in use"

    oom=$(count_oom); oom=${oom:-0}
    (( oom > last_oom )) && echo "WARN $((oom - last_oom)) new OOM kill(s) in dmesg"
    last_oom=$oom

    # containers that died on their memory limit
    dead=$(docker ps -a --filter "label=qcurious.owner" --filter "status=exited" \
           --format '{{.ID}}' 2>/dev/null | head -20 | xargs -r docker inspect \
           --format '{{.State.OOMKilled}}' 2>/dev/null | grep -c true | tr -dc '0-9')
    dead=${dead:-0}
    (( dead > 0 )) && echo "WARN $dead qcurious container(s) OOM-killed (raise DOCKER_MEMORY)"

    for port in 8000 8001; do
        curl -s -m 5 "http://localhost:$port/v1/models" >/dev/null 2>&1 || \
            echo "WARN vLLM on :$port not responding"
    done

    t0=$(date +%s%N)
    timeout 60 docker run --rm --entrypoint true curiositybench:latest >/dev/null 2>&1
    ms=$(( ($(date +%s%N) - t0) / 1000000 ))
    (( ms > 15000 )) && echo "WARN docker start latency ${ms}ms (daemon saturated)"

    sleep "$INTERVAL"
done
