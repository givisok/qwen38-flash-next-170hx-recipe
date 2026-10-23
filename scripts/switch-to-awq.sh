#!/usr/bin/env bash
set -euo pipefail
# Switch the serving stack to the leoncca AWQ-g32 checkpoint.
# Run AFTER the CT restart (swap:0 applied) and download verification.
cd /root/qwen38fn
docker rm -f qwen-intel 2>/dev/null || true
rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
# KV fs tier is content-addressed by TOKENS; cached values belong to the old
# checkpoint — wipe or the new model would read stale KV blocks.
rm -rf /home/models/kv-disk/* 2>/dev/null || true
MODEL_DIR=/home/models/qwen38fn-awq-g32 \
CONFIG=/root/qwen38fn/configs/awq-1m-fp8-ssmbf16.yml \
IMAGE=vllm-sm80:q38fn FORK=/root/src/vllm-sm80 \
PP_PARTITION=17,16,15 ./run-qwen38fn.sh
# ensure boot persistence whenever this script relaunches the container
docker update --restart unless-stopped qwen-intel >/dev/null
