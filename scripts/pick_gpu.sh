#!/usr/bin/env bash
# Print the first GPU (default candidates: 7 6) that is idle; exit 1 if none. Never 0/1 (bv4_smoke containers).
set -euo pipefail
for g in "${@:-7 6}"; do for i in $g; do
  case $i in 0|1) echo "refusing GPU $i" >&2; continue;; esac
  IFS=', ' read -r mem util uuid < <(nvidia-smi -i "$i" --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits)
  [ "$mem" -lt 100 ] && [ "$util" -eq 0 ] || { echo "GPU $i busy: mem=$mem util=$util" >&2; continue; }
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then echo "GPU $i has a process" >&2; continue; fi
  if docker ps -q | xargs -r docker inspect --format '{{json .HostConfig.DeviceRequests}}' | python3 -c '
import json,sys
i,u=sys.argv[1:]
for l in sys.stdin:
    for r in json.loads(l) or []:
        ids=r.get("DeviceIDs") or []
        if i in ids or u in ids or r.get("Count")==-1: sys.exit(0)
sys.exit(1)' "$i" "$uuid"; then echo "GPU $i assigned to a running container" >&2; continue; fi
  python3 - "$i" <<'P' || { echo "GPU $i held by peer queue" >&2; continue; }
import sqlite3,sys
c=sqlite3.connect('file:/data1/j50057870/gpuq/ledger.db?mode=ro',uri=True)
held={g for (gs,) in c.execute("select gpus from tasks where host='37' and status='running'") for g in (gs or '').split(',')}
pend=c.execute("select count(*) from tasks where status not in ('done','failed','cancelled')").fetchone()[0]
if pend: print('warning: peer queue has',pend,'active/pending tasks',file=sys.stderr)
sys.exit(sys.argv[1] in held)
P
  echo "$i"; exit 0
done; done
exit 1
