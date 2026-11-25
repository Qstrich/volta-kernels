#!/usr/bin/env bash
# All step-1 suites on one GPU, sequentially; records append to results.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."; export STEP=step1 GPUS=${GPUS:-7}
for s in "backends bench/backends.py" "ceil bench/ceilings.py" "sdpa bench/sdpa.py full mem_efficient,math" "block bench/profile_block.py"; do
  set -- $s; n=$1; shift
  echo "### $n $(date -u +%T)"; scripts/run.sh $n python "$@" 2>&1 | grep -v '"suite": "backends"' ; sleep 5
done
