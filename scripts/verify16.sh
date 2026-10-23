#!/usr/bin/env bash
# 16-parallel throughput test + GPU utilization sampling.
set -u
N=${N:-16}
TOK=${TOK:-900}
TAG=${TAG:-dense}
OUT=/root/verify-$TAG
mkdir -p "$OUT"

for i in $(seq 1 "$N"); do
 (
  s=$(date +%s.%N)
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8\",\"max_tokens\":$TOK,\"temperature\":1.0,\"top_p\":0.95,\"messages\":[{\"role\":\"user\",\"content\":\"Перечисляй факты об океанах подряд. Серия $i.\"}]}" \
    > "$OUT/req$i.json"
  e=$(date +%s.%N)
  echo "$s $e" > "$OUT/req$i.time"
 ) &
done

# GPU sampling: 8 rounds x 2s starting 6s in
sleep 6
for r in $(seq 1 8); do
  nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader
  echo "---"
  sleep 2
done > "$OUT/gpu.log" 2>&1 &

wait
echo "=== PER-REQUEST (start end n_tok tok/s) ==="
python3 - "$OUT" "$N" <<'PYEOF'
import json,sys
out,n=sys.argv[1],int(sys.argv[2])
tot_tok=0.0; tot_dur=0.0; first_s=None; last_e=0.0; sps=[]
for i in range(1,n+1):
    try:
        t=open(f"{out}/req{i}.time").read().split()
        s,e=float(t[0]),float(t[1])
        d=json.load(open(f"{out}/req{i}.json"))
        nt=d.get("usage",{}).get("completion_tokens",0)
        if d.get("error"): print(f"req{i}: ERROR {str(d['error'])[:120]}"); continue
        dur=e-s; tot_tok+=nt; sps.append(nt/dur)
        first_s=s if first_s is None else min(first_s,s); last_e=max(last_e,e)
        print(f"req{i}: {nt} tok in {dur:.1f}s = {nt/dur:.1f} tok/s")
    except Exception as ex: print(f"req{i}: FAIL {ex}")
wall=last_e-first_s if first_s else 0
print(f"=== AGGREGATE: {tot_tok:.0f} tok / {wall:.1f}s wall = {tot_tok/wall:.1f} tok/s (sum of per-req: {sum(sps):.1f}) ===")
PYEOF
echo "=== GPU SAMPLES (idx util% power mem) ==="
grep -v '^---' "$OUT/gpu.log" | head -24
echo "=== ENGINE STATS ==="
docker logs qwen-intel 2>&1 | grep -E "Engine 000|generation throughput" | tail -4
