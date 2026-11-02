# Qwen3.8-Flash-Next (AWQ-g32, Uncensored) on 3× CMP 170HX — the full recipe

> Main version: English. Russian version: [`README.ru.md`](README.ru.md).

Home LLM rig: **Qwen3.8-Flash-Next 180B-A6B (Uncensored, AWQ-g32)** on vLLM with
pipeline parallelism across 3 **CMP 170HX cards (unlocked to 64 GB, sm80/Ampere)**,
1M-token context, MTP speculative decoding, fp8 KV cache.

Working point (2026-11-02): **production = MTP-4** (k=4 + bf16 mamba state,
`gpu-memory-utilization: 0.95`, KV pool **5.28M tokens / 5.04×**; NOTES §19–21).
The retained **DFlash k5 lane is now full-1M** — its YaRN-boundary wall
(acc 0.01, ~47 tok/s past 262k) turned out to be the drafter's RoPE table
running out of rows, fixed by one config line on 2026-11-02 (see the
boundary note in §4): **1.79 acc/draft and 141.7 tok/s @1M** — above
MTP-4's 134 — with every shallow gate unchanged. Below ~250k it is still
faster on code than MTP-4 (sparkDash code 253–257 vs 195 tok/s solo,
N=8 code 832 vs 686); one command away (`SPEC_MODE=dflash`).

Why MTP-**4** is the base and not the old MTP-3 prod: same-day A/B put k=4
strictly ahead at the same pool class (5.16M vs 5.21M, both bf16 mamba) —
sparkDash solo code **195 vs 169**, structured **241 vs 206**, prose parity
149 vs 150 — and the extra draft token costs nothing at depth (1.63
acc/draft @1M, no cliff). MTP-3 (base yml `awq-1m-fp8-ssmbf16.yml`) is
kept as a deeper rollback rung only; nothing runs it today.

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

Final serve config: [`configs/awq-1m-fp8-ssmbf16-mtp4.yml`](configs/awq-1m-fp8-ssmbf16-mtp4.yml)
(MTP-4 production lane; the launcher's plain `./run-qwen38fn.sh` defaults to it since
2026-11-02). [`configs/awq-1m-fp8-ssmbf16.yml`](configs/awq-1m-fp8-ssmbf16.yml) (MTP-3)
is the deeper rollback rung. The endpoint requires an API key (`api-key:` in the yml,
set on the server).

| Parameter | Value | Why |
|---|---|---|
| `pipeline-parallel-size` | 3 | P2P between the cards is broken (DMA glitches on this hardware) → TP is impossible, PP only |
| `VLLM_PP_LAYER_PARTITION` | **17,17,14** | 48 layers. The last rank carries the target `lm_head` + MTP draft + the W4A16 draft head, and the KV planner takes the **min raw bytes across ranks** — PP2 binds the pool. 2026-10-30: one layer off PP2 → pool 4.69×→4.97×, decode neutral (max layers/rank still 17), N=6 aggregate record 402 tok/s. Earlier: 17,16,15 (utilization sweep, +7.5% single-stream vs 16/16/16); 17,18,13 NOT recommended — hits PP0's byte ceiling and unbalances decode |
| `speculative-config` | `method: mtp, model: …/mtp-fp8, num_speculative_tokens: 4` | +75% single-stream (120 vs 66–70 without MTP); ngram_gpu tried — worse. 2026-10-28: switched to the author's FP8 draft (`mtp-fp8/` in the checkpoint dir) — one config line, +8% single-stream, KV pool 4.29×→4.69×. Depth 4 was rejected on 2026-10-28 (−8% on MTP-3-era stack), then re-won on 2026-11-01 at the bf16-mamba state (solo code 195 vs 169, structured 241 vs 206, no depth cost to 1M) → production |
| `max-num-seqs` | 16 | 8→16 raised aggregate from 360 to ~500–600 tok/s; cards sit at 84–100% utilization |
| `cudagraph_capture_sizes` | `[1,2,4,...,64]` step 4 | from the mannydelrio repo: the MTP batch rounds up to 4 |
| `kv-cache-dtype` | fp8_e4m3 | software emulation on sm80; KV pool of 5.28M tokens (current) |
| `max-model-len` | 1,048,576 | YaRN ×4 over 262144 (in `hf-overrides`, plus `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`), `block-size: 1728`, `rope_theta 1e7` |
| `layer_types` override | 3× linear_attention + 1× qwen_sparse_attention ×12 | bridges the transformers rename (full_attention → indexed_attention) |
| `NCCL_P2P_DISABLE=1` + `disable-custom-all-reduce` | | P2P hangs on these cards |
| `gpu-memory-utilization` | **0.95** | 0.94 held from bring-up to 2026-11-01 (pool 5.16M/4.92× on MTP-4). 2026-11-02: 0.94→0.95, A/B'd with llm-assay `ab-deploy` (paired seed, no regressions; `runs/gmu095/`) → pool 5,163,367→**5,282,636 tokens, 5.04×** — first time past 5×. 131k battery: cold prefill 4179 tok/s, follow-up turn TTFT 2.10 s; needle 120k PASS |
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
6. **`pp-int8-hop.diff`** (2026-10-28, port of
   [g4bwy](https://github.com/g4bwy/qwen3.8-flash-next-cmp-170hx) patch 0006) —
   int8 PP pipeline handoff (`PP_INT8_HOP=1` in the launcher): cold 120K prefill
   **+15–21%**, decode unchanged, KV pool +13K (halved receive buffers). Gates:
   needle through the int8 path OK, greedy A/B 2/3 byte-identical.
7. **`mtp-draft-head-w4a16.diff`** (2026-10-28, port of
   [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm) 14457f89) +
   `scripts/quantize_lm_head.py` — the MTP draft keeps its own W4A16 GPTQ-Marlin
   `lm_head` (int4 g128 sym, built from THIS checkpoint's `lm_head.weight`;
   artifact `draft-lm-head-w4a16-g128.safetensors` next to the model) instead of
   adopting the target's BF16 head: ~0.95 GiB less weight read per draft step.
   Acceptance unchanged; `MTP_DRAFT_HEAD=1`.
8. **`fn12-bounded-prefill.diff`** (2026-10-28, port of AlexBrandes/vllm 31208d20)
   — bounded prefill beside running decodes (`MIXED_PREFILL_CHUNK=864`): decode
   stalls >1 s during a concurrent 120K prefill went **37–38 → 0** (max gap
   1.19 s → 0.75 s) at +10% prefill TTFT. 512 and 1024 measured and rejected.
9. **`pp-head-missing.diff`** (2026-10-30, after
   [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) f2815b1
   / newer upstream) — non-last PP stages build `PPMissingLayer` instead of a full
   BF16 `lm_head`: −1.17 GiB of dead weight on each of PP0/PP1 (also: no dead-head
   warmup, faster checkpoint load). Note: the KV pool itself is set by the
   raw-byte minimum across ranks (a per-layer-normalized estimate of the pool gain
   was wrong — see TUNING); the pool gain came from the §3 repartition instead.
10. **`qwen4exp-dflash2-target.diff` +
    `configs/awq-1m-fp8-ssmbf16-dflash.yml`** (2026-10-31 port → 2026-11-01
    two fixes → **current production lane: DFlash 2 k=7**) — port of the
    inco.ai DFlash 2 drafter target-side via ElectroHobo's recipe.
    `Qwen4ExpModel` gains `EagleModelMixin` + aux-state capture (completed HC
    state entering layers 6/16/26/36/46, hc-mean — exactly what inco.ai
    trained the drafter on); the states ride IntermediateTensors across PP
    ranks; the config guard accepts `dflash`.

    **Fix 1 — pool collapse (2026-11-01).** The draft's SWA block (1728 tok ×
    4096 B/tok bf16 KV × 5 layers = 35.4 MB) quadrupled the pool's
    bytes-per-block on the drafter's rank: num_blocks 3492→939, pool 1.45× of
    1M. Fix: `VLLM_SWA_BLOCK_CAP=432` — the 5-layer draft group page becomes
    8,847,360 B == the GDN group page, pool block stays at the GDN level.
    k3: pool 5.24M / 5.00×; k7: 4.01M / 3.83×.

    **Fix 2 — prefix-cache regression (2026-11-01, found via a live pi
    session).** The draft SWA group made `partial_hash_hits_enabled` return
    False (SlidingWindowManager has no fine-grained lookup), dropping the
    whole engine to block-aligned prefix hits; block-aligned Mamba lookups
    cannot see the partial/checkpoint state hashes align-mode GDN exports, so
    every first resend of a prompt shape re-prefilled from scratch (11–12 s at
    45k context — agents pay it on every turn). Fix:
    `VLLM_SWA_NO_PREFIX_CACHE=1` — the drafter group opts out of prefix
    caching (its 2048-token window KV is recomputed on the resumed tail);
    fine-grained hits stay enabled for the target. Verified:
    `scripts/cache_probe.py` — identical resend 0.26–1.5 s, grown-head resend
    1.59 s (was 11.7 s), grown-tail resend 0.27 s; live pi session cacheRead
    growing to 123k tokens over 28 turns.

    **Lane config.** PRODUCTION: k=5 + `mamba-ssm-cache-dtype: bfloat16` at
    block 1728, chunk 6912 — the bf16 GDN slot (221 KB) lets 6 slots fit the
    1728 page (fp32 maxes at k3; k4 = 2.21 MB > the 1.77 MB page, assert), so
    the pool stays in the MTP-4 band. k7 variant: block 3456 (1+7 fp32
    slots), chunk 10368, schedule `[[1,2,7],[3,4,5],[5,16,3]]`, pool 4.01M.
    All lanes: `attention_backend: FLASH_ATTN` for the drafter (auto-picked
    FlashInfer drops the drafter rank to cudagraph NONE = 3× slower) and
    draft-only `kv_cache_dtype: auto` (FLASH_ATTN rejects fp8 KV on SM80; the
    target keeps fp8_e4m3).

    **Numbers (2026-11-01, sparkDash DecodeBench protocol replica —
    `scripts/sd_bench.py`; per-request medians, real tokens from usage).**

    | lane | solo code / prose / structured | N=8 code agg | pool |
    |---|---|---|---|
    | MTP-3 | 169 / 150 / 206 | 648 | 5.21M |
    | MTP-4 | 195 / 149 / 241 | 686 | 5.16M |
    | dflash k=3 † | 197 / 158 / 206 | 696 | 5.24M |
    | dflash k=7 † | 292–299 / 152–181 / 355–360 | 707–741 | 4.01M |
    | **dflash k=5 + bf16 mamba** † | **253–257 / 173 / 285** | **832** | **5.19M** |

    † dflash depth limit **removed 2026-11-02**: the drafter's RoPE table had
    only 262,144 rows while the speculator feeds real target positions to 1M —
    past the boundary the reads ran off the table (silent garbage angles,
    acceptance 0.01). One config line (drafter `max_position_embeddings`
    262144→1048576, rope untouched) restores **1.79 acc/draft at 512k and 1M**
    with every shallow number unchanged (see the boundary note below and
    `runs/dflash-1m/`).

    2026-11-02: production MTP-4 raised to `gpu-memory-utilization: 0.95`
    → pool **5.28M / 5.04×** (llm-assay A/B, no regressions; `runs/gmu095/`).

    k5+bf16 quality gates: needle 64k and 120k PASS (`scripts/needle_deep.py`),
    prefix cache at 95k = 0.41 s resend (95,584 / 95,600 tokens cached),
    decode flat at 128k depth (code 269 tok/s), acceptance 60.6% / 4.03 tok/step.
    2026-11-02 re-run after the rope-table fix: every gate reproduced
    (solo 255/173/284, acceptance 60.3/4.01, code @128k 268) and depth now
    flat through 1M — code 300 @512k, prose 163.6 @128k → 157.1 @512k,
    needle 1M again 18/18.

    Full-depth gate (`scripts/needle_1m.py`, 2026-11-01): one 1,010,326-token
    record wall, six needles at 5/20/40/60/80/95 % depth, each asked 3× with
    different phrasings — **18/18 PASS**, exact codes, no cross-needle leaks
    (negative control on an absent station came back invented, not leaked).
    Cold prefill 1.01M = 235.6 s = 4,287 tok/s; every resend rode the prefix
    cache at 1M scale — 1,007,424 / 1,010,360 tokens cached, TTFT 4.3–4.5 s
    (~52× vs cold). This is also the SWA cache fix proven at full depth.

    **Boundary caveat (NOTES §19–20) — RESOLVED 2026-11-02**: the drafter
    shipped `max_position_embeddings: 262144, rope_type: default`, so its
    cos/sin table had exactly 262,144 rows while the speculator feeds real
    target positions up to 1M — past the boundary both draft queries and the
    injected target-state keys read **past the end of the table** (Triton
    garbage angles, silently: acc 0.01, decode pinned at ~47 tok/s,
    identical at 512k and 1M). The first diagnosis ("no rope_scaling /
    yarn mismatch") was wrong in the mechanism but right in the location.
    Fix: drafter config.json `max_position_embeddings: 262144 → 1048576`
    with `rope_parameters` **untouched** — RoPE attention depends only on
    relative position, so angles below 262,144 are bit-identical (all
    shallow gates unchanged) and the ×4 table costs no pool (5,185,037 vs
    5,182,071). Do **not** copy the target's yarn block into the drafter:
    yarn blends frequencies at all positions and would break the working
    ≤256k range too (the drafter was trained on plain theta-1e4 rope).
    Post-fix battery (`runs/dflash-1m/battery-20261102.txt`): acc/draft
    1.62 @250k control / **1.79 @512k / 1.79 @1M**, decode 121 / 131.8 /
    **141.7 tok/s** (MTP-4 prod: 134 @1M); code @128k 268 / @512k 300;
    needle 120k PASS; needle 1M 18/18 PASS; battery acceptance 60.3% /
    4.01 (gate 60.6 / 4.03); sparkDash solo 255 / 173 / 284. Production
    stayed on MTP-4 through this fix — switching lanes is a separate
    decision. The MTP head needs none of this: it shares the target's rope
    and was always flat to 1M (1.63 acc/draft, 134 tok/s).

    Under load (N=16/32) all lanes sit in one ~2.8–3.4k chars/s band —
    speculation gains die with batch; the lanes differ at light load, where
    k7 dominates (+50–75 % over MTP solo). Decode at depth is flat 32k→128k
    (code 288–294 tok/s on k7). Prefill exact (prompt_tokens-based): 3.66k
    @13k, 4.35–4.41k @54–110k — parity with MTP. Needle 64k PASS.

    Production state: fork `q38fn/stack-local` @ `3c367a58e5` (content =
    branch `dflash2-pool` @ `9eadcf0a8f`; increment diff in
    `patches/dflash2-pool-increment.diff`), kit `main` @ `f248264`. Both
    lanes boot-tested from this tree. Diagnostics behind `VLLM_CACHE_DIAG=1`.
    Rollback ladder: k3 = block 1728/chunk 6912/k3 (kit history 0a95834);
    MTP-4 = the plain default since 2026-11-02 (`./run-qwen38fn.sh`,
    `SPEC_MODE=mtp` explicitly works too; earlier state `prod-mtp4-20261101`).
    Drafter: `incoai/Qwen3.8-Flash-Next-DFlash2` (Apache-2.0, 1.1 GB) at
    `/home/models/qwen38-dflash2-drafter`. Tag: `prod-dflash-k7-20261101`.

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

## 7. Measurements

Bring-up era (2026-10-24…26, AWQ-g32 vs Intel AutoRound W4A16):

| Test | Intel W4A16 | AWQ-g32 |
|---|---|---|
| Single stream (TG) | 115–121 tok/s | 117–129 tok/s |
| 16 concurrent, per request | 45.4 | 44.9–45.0 |
| 16 concurrent, aggregate | 561 | 514–604 (generation-length noise) |
| Cold prefill @ 79K tokens | 7,436 tok/s | 7,880–7,920 tok/s |
| Echo (text copying) | 138.2 tok/s | 148.4 tok/s |
| GPU KV pool | 4.45M | 4.57M tokens |
| Checkpoint on disk | 169 GB | 133 GB (PLE 51 GB fp8) |

Current state after the 2026-10-27…30 tuning series (prefix-cache knobs, LPT 2048,
int8 hop, W4A16 draft head, mtp-fp8 draft, FN12 bounded prefill, PPMissingLayer
head fix, PP repartition — every step with A/B data in
[`TUNING-2026-10-27.md`](TUNING-2026-10-27.md)):

| Test | bring-up era | current (2026-10-30) |
|---|---|---|
| Single stream, battery mean (4×2, temp 0) | 117–129 | **132** (105–151; 150-ceiling above every prior cell) |
| 6 concurrent, aggregate | — | **375 / 402 tok/s** (402 = best ever recorded) |
| Cold prefill 120K | ~3.7k tok/s | 3.7–4.5k tok/s (int8 hop +15–21%) |
| GPU KV pool | 4.57M / 4.36× | **5,214,494 / 4.97×** |
| MTP acceptance | 2.35–2.6 tok/step | 2.35–2.75 tok/step |

Production lane (2026-11-01 final, DFlash 2 k=5 + bf16 mamba state — sparkDash
DecodeBench protocol, `scripts/sd_bench.py`; cache from `scripts/cache_probe.py`):

| Test | MTP-4 lane | DFlash 2 k=7 | **DFlash 2 k=5+bf16 (prod)** |
|---|---|---|---|
| Solo code / prose / structured (tok/s) | 195 / 149 / 241 | 292–299 / 152–181 / 355–360 | **253–257 / 173 / 285** |
| N=2/4 code, per-request | 159 / 138 | 245 / 228 | 213 / 175 |
| N=8 code / structured aggregate | 686 / 1119 | 741 / 976 | **832 / 1268** |
| Solo battery (py/rust/edit/sql) | 151 / 179* / 170* / 281* | 160 / 199 / 223 / 371 | 157 / 189 / 205 / 358 |
| Prefix-cache resend, 45k / 95k | 0.26–0.89 s / — | 0.26–1.50 s / — | 0.9 s / **0.41 s** |
| Decode at 32k / 128k depth, code | — | 294 / 288 (flat) | — / 269 (flat) |
| Needle | 64k PASS | 64k PASS | **64k + 120k PASS** |
| Cold prefill (exact, prompt_tokens) | — | 3.66k @13k, 4.35–4.41k @54–110k | same class |
| GPU KV pool | **5.28M / 5.04×** (0.95, 11-02) | 4.01M / 3.83× | 5.19M / 4.94× |

(*MTP-4 battery single-run, others 2-round medians.)

Live pi-agent session same-day A/B (pi counts output tokens over full turn
time, prefill included): MTP stretch median ~88 out tok/s (~130 with
reasoning), k7+fix stretch ~100 (~170, peaks 186/293).

Benchmark scripts: `scripts/verify16.sh` (parallel load + GPU utilization),
`scripts/prefill.py` (PP/TG), `scripts/oomtest.py` (memory stress).
All `scripts/*.py` probes take the endpoint key from the `QWEN38FN_API_KEY`
env var and the base URL from `QWEN38FN_BASE` (default `http://localhost:8080`,
i.e. run them on the CT itself); no secrets live in the repo.
the tuning-series battery: `scripts/bench_draft_ab.py`, `scripts/bench_conc.py`,
`scripts/probe_contention.py` (artifacts in `runs/`).

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
