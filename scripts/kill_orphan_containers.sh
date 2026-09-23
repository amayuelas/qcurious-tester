#!/bin/bash
# Kill test containers whose owning runner process is gone.
#
# DockerCoverageRunner labels every container qcurious.owner=<pid> and cleans
# up on normal exit or SIGTERM. A runner that is SIGKILLed or OOM-killed can't,
# so its in-flight containers keep running (and holding memory). This sweep
# kills only containers whose owner PID no longer exists, so it is safe to run
# while other experiments are live.
set -uo pipefail

docker ps --filter "label=qcurious.owner" \
    --format '{{.ID}} {{.Label "qcurious.owner"}}' |
while read -r id owner; do
    if ! kill -0 "$owner" 2>/dev/null; then
        docker kill "$id" >/dev/null 2>&1 && echo "killed $id (owner $owner gone)"
    fi
done
