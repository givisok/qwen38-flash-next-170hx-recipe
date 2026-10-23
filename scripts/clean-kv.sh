#!/usr/bin/env bash
# Wipe the persistent NVMe KV tier and restart the engine.
# Use when answers turn to garbage: the fs-tier can hold corrupted block
# entries written under eviction churn (race in the patched LRU manager).
# Cost: cold prefix cache (~1-2 min of re-prefill) + ~8 min restart.
set -euo pipefail
docker restart qwen-intel >/dev/null &
P=$!
rm -rf /home/models/kv-disk/*
wait $P
sleep 2
docker update --restart unless-stopped qwen-intel >/dev/null
echo "kv-disk wiped, engine restarted (~8 min to ready). Watch: docker logs -f qwen-intel | grep 'startup complete'"
