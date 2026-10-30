# Qwen3.8-Flash-Next (AWQ-g32, Uncensored) on 3× CMP 170HX — the full recipe

> Main version: English. Russian version: [`README.ru.md`](README.ru.md).

Home LLM rig: **Qwen3.8-Flash-Next 180B-A6B (Uncensored, AWQ-g32)** on vLLM with
pipeline parallelism across 3 **CMP 170HX cards (unlocked to 64 GB, sm80/Ampere)**,
1M-token context, MTP speculative decoding, fp8 KV cache.

Working point: ~120 tok/s single-stream, ~45 tok/s per request at 16 concurrent
(~600 tok/s aggregate), prefill up to 7,900 tok/s on a 79K-token prompt.

---

## 1. Sources — what was taken from where

| Component | Source | What was taken |
|---|---|---|
| Base stack (vLLM fork + docker image + launcher) | [clee/qwen3.8-flash-next-3x170hx](https://github.com/clee/qwen3.8-flash-next-3x170hx) | The `vllm-sm80` fork (the `q38fn/stack` line on top of the pinned nightly `7d0b4e57`), the launch script, PLE streaming from NVMe (preadv), YaRN override, KV tiering (later disabled — see §6) |
| Model (checkpoint) | [leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) (~137 GB, 87 shards) | AWQ W4A16 **group-32** (group size 32 chosen to fit the 640-wide experts: 640/4 ranks = 160, divisible by 32 but not by 128), damped GPTQ on 169K rows of real activations, KL 0.0214 / top-1 97.08% vs BF16; PLE table in **fp8** (51 GB instead of 95 GB BF16); calibrated QSA KV scales; MTP draft included; uncensored base is [orcarouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) |
| Tuning for the same model family | [mannydelrio/qwen38-flash-next-2x-cmp170hx](https://github.com/mannydelrio/qwen38-flash-next-2x-cmp170hx) | The idea of explicit **cudagraph capture sizes at stride 4** up to 64 (vLLM rounds the MTP batch up to (1+3)=4 tokens; without this, the tail of a 16-stream decode runs without graphs). Their INT4-PLE (tinybeast plugins) is unportable and unnecessary: PLE already streams from NVMe |
| Previous model (for comparison / rollback) | [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | Ran without patches; larger (169 GB), PLE in BF16, slower prefill |
| Chat template | the leoncca checkpoint (`chat_template.jinja`) | thinking mode, reasoning effort **xhigh/medium/low** (any other value → the template rejects with 400) |

Alternatives evaluated and rejected: NVFP4 quant (lambsea) — requires Blackwell;
GGUF/MLX/EXL3 — not vLLM; Mochgolf W4A16-INT8PLE — same size, quality unmeasured,
only proven in someone else's SGLang fork.

## 2. Hardware and container

- Proxmox VE host with 3× NVIDIA CMP 170HX (unlocked to 64 GB = sm80), 32 GB RAM, 2 TB NVMe.
- LXC container **CT 101** (Debian, unprivileged, `nesting=1 keyctl=1`):
  - memory 26000, **swap: 0** (swapping only hurts inference: the tmpfs/KV tier
    starts spilling to disk; see §6 "swap"),
  - passthrough of `/dev/nvidia0-2`, `nvidiactl`, `nvidia-uvm*`, `lxc.prlimit.memlock: unlimited`,
  - `onboot: 1`, 1.4 TB disk.
- Inside: docker, image `vllm-sm80:q38fn` (built from the clee repo's Dockerfile on
  top of the pinned nightly), the fork tree in `/root/src/vllm-sm80`, models in
  `/home/models` (mounted into the container read-only as `/models`).

## 3. Key configuration decisions

Final serve config: [`configs/awq-1m-fp8-ssmbf16.yml`](configs/awq-1m-fp8-ssmbf16.yml).

| Parameter | Value | Why |
|---|---|---|
| `pipeline-parallel-size` | 3 | P2P between the cards is broken (DMA glitches on this hardware) → TP is impossible, PP only |
| `VLLM_PP_LAYER_PARTITION` | **17,16,15** | 48 layers; per-rank utilization/power measurements showed: the last rank also carries the MTP draft — it is the heaviest, so it gets fewer layers. Sweeping 16/16/16 → 17/16/15 gave +7.5% single-stream; 18/16/14 is worse |
| `speculative-config` | `method: mtp, num_speculative_tokens: 3` | The draft ships in the checkpoint; +75% single-stream (120 vs 66–70 without MTP). Tried ngram_gpu — worse |
| `max-num-seqs` | 16 | 8→16 raised aggregate from 360 to ~500–600 tok/s; cards sit at 84–100% utilization |
| `cudagraph_capture_sizes` | `[1,2,4,...,64]` step 4 | from the mannydelrio repo: the MTP batch rounds up to 4 |
| `kv-cache-dtype` | fp8_e4m3 | software emulation on sm80; KV pool of 4.57M tokens |
| `max-model-len` | 1,048,576 | YaRN ×4 over 262144 (in `hf-overrides`, plus `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`), `block-size: 1728`, `rope_theta 1e7` |
| `layer_types` override | 3× linear_attention + 1× qwen_sparse_attention ×12 | bridges the transformers rename (full_attention → indexed_attention) |
| `NCCL_P2P_DISABLE=1` + `disable-custom-all-reduce` | | P2P hangs on these cards |
| `gpu-memory-utilization` | 0.94 | ceiling without OOM; KV pool of 4,573,327 tokens, 4.36× concurrency at 1M |
| **kv-transfer-config (tiering)** | **ABSENT (removed)** | see §6 — the source of KV corruption |

## 4. Fork patches for this checkpoint (required)

They live in [`patches/`](patches/) and are baked into the `/root/src/vllm-sm80`
tree (the launcher bind-mounts the fork files over the image):

1. **`mtp-weights-remap.diff`** — the checkpoint ships QSA KV scales
   (`mtp.layers.0.self_attn.{q,k,v}_scale`). AutoWeightsLoader runs them through
   `quant_config.get_cache_scale_mapper()`, which rewrites `X.k_scale` to
   `X.attn.k_scale` (the stock-attention layout) — QSAAttention has no `attn`
   child → crash "no module or parameter named 'attn'". Fix: k/v go to the
   owner's buffers (`_k_scale`/`_v_scale`), q_scale is dropped.
2. **`ple-fp8-staging.diff`** — PLE staging was hardcoded to BF16 rows
   (`head_dim*2` bytes). The fp8 table has `head_dim*1` = 160-byte rows → a
   320-vs-160 crash on the first warmup. Fix: buffer size derived from the table
   dtype; in forward — a view + elementwise fp8→bf16 dequant (a
   cudagraph-capturable kernel).
3. **`VLLM_PLE_SYNC=1`** (env in the launcher) — the fork's own code suspected a
   race ("to isolate whether the corruption is a staging race"): the async H2D
   copy of PLE rows from pinned memory against the start of forward. Moving the
   table to fp8 shifted the timings — the race opened up. Sync closes it.
4. **`qsa-topk-tiefix.diff`** (2026-10-24, port from
   [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere))
   — canonical top-k for the QSA indexer. The stock `persistent_topk` resolves
   exact score ties at the k-th boundary by atomic arrival order → **the set of
   readable context blocks depends on timing, not on scores** (reproduced on our
   GPUs: 8/8 rows changed their selection across 30 calls). The fix repairs only
   the disputed boundaries (ties → lower indices) and sorts each row ascending in
   the same kernel launch. Launcher knobs: `QSA_CANONICAL=1` / `QSA_SORTED=1` (on
   by default; rollback via `QSA_CANONICAL=0 QSA_SORTED=0 ./switch-to-awq.sh`
   without a rebuild). No speed regression (A/B: ON 113 vs OFF 106 tok/s median —
   sorted indices give better KV read locality). Details and tests:
   `patches/NOTES.md`, `patches/qsa-tiefix-gpu-test.py`.
5. **`qwen4-precision.diff` + `qwen4_precision.py`** (2026-10-24, port of the same
   Morrowmake precision switches) — **router logits FP32 enabled**: GateLinear
   tier 5 (cuBLAS bf16×bf16→fp32, works on sm80 too) gives better score separation
   and fewer exact ties at top-10 — routing closer to the reference. A/B:
   acceptance 50.4% vs 51.1% baseline (within boot noise), speed unchanged. The
   second switch (per-stream MTP norm) — **disabled**: acceptance dropped to
   48.4%; the draft is calibrated with the flat norm. Additionally:
   `VLLM_MOE_SKIP_PADDING=1` (padding rows never reach the experts) is already on
   by default in the fork — no port needed. Details: `patches/NOTES.md`.

## 5. Running and operations

```bash
# on CT 101, from /root/qwen38fn
cd /root/qwen38fn && IMAGE=vllm-sm80:q38fn FORK=/root/src/vllm-sm80 ./run-qwen38fn.sh   # start
./switch-to-awq.sh    # model switch: rm container, clean /dev/shm + kv-disk, start, restart-policy
./clean-kv.sh         # the KV-corruption remedy (before tiering was disabled): wipe kv-disk + restart
```

- Docker RestartPolicy — **unless-stopped** (otherwise the server did not come
  back after a host power-off — the main cause of "it died after a reboot").
- The launcher cleans `/dev/shm/vllm_offload_*.mmap` and psm_* left over from a
  previous kill — an unremoved 8 GB mmap blocks CPU-tier allocation.
- Endpoint: `http://<ct-ip>:8080/v1`, the model is served as `qwen3.8`. Clients:
  the pi agent (`~/.pi/agent`) and Hermes Agent (CT 102, `reasoning_effort:
  medium` in its `config.yaml`; only xhigh/medium/low are allowed — the template
  rejects anything else).

## 6. Pitfalls we stepped on (and their fixes)

1. **Swap in LXC**: the CT's swap was pegged at 100% (4 GB) — KV/tmpfs spilled to
   disk, latency died. Fix: `pct set 101 --swap 0` + `vm.swappiness=10` on the
   host + `swapoff -a && swapon -a` to drain it. Inference does not need swap.
2. **"Word salad" — multilingual token soup** (`sm exit asset 资产...`):
   - NOT the model, NOT the template, NOT reasoning_effort. Different suspects as
     the investigation went: at first it looked like effort=medium (a masking
     effect: different effort → different prefix hashes → different blocks in the
     tier), then like the PLE race (a real one, closed with sync).
   - **Final root cause: KV offload tiering** (OffloadingConnector: 8 GB CPU + NVMe
     fs tier) — the fork's code returned corrupted blocks for multi-chunk cached
     prefixes. Proven by ablation: the same prompt — 5/5 garbage with tiering,
     6/6 clean without, then 0/32 garbage in an 8-thread battery without tiering.
   - Tiering **removed from the config** (the block text is preserved in
     `patches/NOTES.md`). The GPU prefix cache stays; decode speed unchanged
     (45.0 vs 44.9 tok/s per request). Lost: the warm prefix cache surviving a
     restart. To bring it back — only by fixing the candidates listed in
     `patches/NOTES.md` (lookup/eviction race in fs/manager.py, chunk offsets in
     submit_store, draft-group exclusion in spec.py).
3. **The MTP draft wouldn't load** on the new checkpoint — patch #1 (§4).
4. **PLE in fp8** didn't fit the staging path — patch #2 (§4).
5. **Wipe the kv-disk when switching models**: the fs tier is addressed by tokens,
   but the values come from the old weights (tiering is off now, but keep the rule
   for the future).
6. **OOM stress does not crash the server**: 8×249K and 5×947K tokens (pool
   oversubscription) — everything queues, oom_kill=0; script `scripts/oomtest.py`.
   But exactly this churn pushed the tier into eviction and triggered item 2.

## 7. Measurements (AWQ-g32, final, vs Intel AutoRound W4A16)

| Test | Intel W4A16 | AWQ-g32 |
|---|---|---|
| Single stream (TG) | 115–121 tok/s | 117–129 tok/s |
| 16 concurrent, per request | 45.4 | 44.9–45.0 |
| 16 concurrent, aggregate | 561 | 514–604 (generation-length noise) |
| Cold prefill @ 79K tokens | 7,436 tok/s | 7,880–7,920 tok/s |
| Echo (text copying) | 138.2 tok/s | 148.4 tok/s |
| GPU KV pool | 4.45M | 4.57M tokens |
| Checkpoint on disk | 169 GB | 133 GB (PLE 51 GB fp8) |

Benchmark scripts: `scripts/verify16.sh` (parallel load + GPU utilization),
`scripts/prefill.py` (PP/TG), `scripts/oomtest.py` (memory stress).

## 8. Reproducing from scratch (short version)

1. Proxmox: create the CT (§2), pass the cards through, `swap 0`, install docker.
2. Take the clee repo (fork + Dockerfile), build `vllm-sm80:q38fn`,
   checkout the `q38fn/stack` branch into `/root/src/vllm-sm80`.
3. Download the leoncca AWQ-g32 checkpoint (137 GB, verify SHA256SUMS) into
   `/home/models/qwen38fn-awq-g32`; chat_template → into the launcher configs.
4. Apply the §4 patches, drop in `configs/` and `scripts/` from this repo.
5. `docker update --restart unless-stopped` after the first start.
6. Verify: single request → N=16 (`verify16.sh`) → prefill (`prefill.py`) →
   long agentic prompts with tools and history (multi-hour runs!).

## 9. Credits

This recipe stands on the work of others — the starting configuration and most
later improvements were taken from their repos/checkpoints:

- **[clee](https://github.com/clee/qwen3.8-flash-next-3x170hx)** — the whole base
  stack: the `vllm-sm80` fork (`q38fn/stack` line), the docker image, the launcher,
  PLE streaming from NVMe, the YaRN override. Without this there would be no build.
- **[leoncca](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32)**
  — the AWQ-g32 checkpoint we serve: group-32 quant, damped GPTQ calibration, fp8
  PLE table, calibrated QSA KV scales, the MTP draft included in it.
- **[orcarouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)**
  — the uncensored base model behind the checkpoint.
- **[Morrowmake](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere)** — the
  source of two ports: the QSA top-k tie fix (`qsa-topk-tiefix.diff`) and the
  Qwen4Exp precision switches (`qwen4-precision.diff` + `qwen4_precision.py`).
- **[mannydelrio](https://github.com/mannydelrio/qwen38-flash-next-2x-cmp170hx)** —
  the cudagraph capture-size lattice (stride 4 up to 64) that lets the tail of
  MTP batches run under graphs.
- **[g4bwy](https://github.com/g4bwy/qwen3.8-flash-next-cmp-170hx)** — the
  prefix-cache knobs in [`TUNING-2026-10-27.md`](TUNING-2026-10-27.md) and the
  int8 PP-handoff patch (their 0006).
- **[AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next)** —
  the scheduler/prefill lane: the W4A16 MTP draft lm_head
  ([`AlexBrandes/vllm@14457f89`](https://github.com/AlexBrandes/vllm/commit/14457f89))
  and the FN12 bounded prefill beside decodes
  ([`AlexBrandes/vllm@31208d20`](https://github.com/AlexBrandes/vllm/commit/31208d20)).
- **[Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)** —
  the model that ran on this hardware before, our comparison baseline.
- **The vLLM project** — the pinned upstream nightly `7d0b4e57` that the fork is
  built on.

Thanks to all of them. Every bug introduced by these ports is ours, not theirs.
