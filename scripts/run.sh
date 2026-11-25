#!/usr/bin/env bash
# run.sh <name> <cmd...> : run on an idle GPU in the project image; checks the GPU before and after.
set -euo pipefail
name=$1; shift
P=$(cd "$(dirname "$0")/.." && pwd); D=/data1/j50057870/volta-kernels
g=$("$P/scripts/pick_gpu.sh" ${GPUS:-7 6})
echo "[run] GPU $g, container volta-$name" >&2
rc=0
docker run --pull=never --rm --name "volta-$name" --gpus "device=$g" --user "$(id -u):$(id -g)" --ipc=host \
  -e HOME=/tmp -e USER=volta -e LOGNAME=volta -e VOLTA_HOST_GPU="$g" -e OUT="$D/${STEP:-step2}" -e PYTHONPATH="$P/ext/build:$P/ext:$P/bench:$P/tests" \
  -v "$P:$P" -v "$D:$D" -w "$P" volta-kernels:cu126 "$@" || rc=$?
sleep 8  # utilization counter lags our own exit
"$P/scripts/pick_gpu.sh" "$g" >/dev/null && echo "[run] GPU $g idle after run" >&2 || { echo "[run] WARNING: GPU $g not idle after run" >&2; }
exit $rc
