# Fork patches required for leoncca AWQ-g32 checkpoint (2026-10-23)

Patches on top of q38fn/stack (bind-mounted by run-qwen38fn.sh):

1. mtp.py — `_remap_mtp_weight_name`: the AWQ-g32 checkpoint ships QSA cache
   scales (`mtp.layers.0.self_attn.{q,k,v}_scale`). AutoWeightsLoader applies
   quant_config.get_cache_scale_mapper() which rewrites `X.k_scale` into
   `X.attn.k_scale` (stock attention layout); QSAAttention has no `attn`
   child and the draft path never runs _remap_qsa_cache_scale_name → crash
   "no module or parameter named 'attn'". Fix: route k/v onto the owner
   buffers (`_k_scale`/`_v_scale`), drop q_scale (never cached).
2. ngram_embedding.py — `_ple_init_staging` + staged forward: staging buffer
   was hardcoded BF16 rows (head_dim*2 bytes). The AWQ-g32 PLE table is
   FP8-e4m3 (head_dim*1 = 160 B/row) → 320-vs-160 copy crash at first
   warmup. Fix: size rows by the table dtype (weight.dtype), view as the
   table dtype and dequantize fp8→bf16 in the staged forward branch
   (elementwise cast, cudagraph-capturable).
3. ops/qsa_indexer.py + ops/qsa_topk_tiefix.py — canonical QSA top-k tie fix,
   ported 2026-10-24 from Morrowmake/vllm-cmp170hx (ampere), see the section
   below and `qsa-topk-tiefix.diff`.
4. precision.py + hooks in model.py/mtp.py — Qwen4Exp precision switches,
   also ported 2026-10-24 from Morrowmake (ampere): `qwen4-precision.diff`,
   `qwen4_precision.py`. Shipped ON: router logits FP32 (GateLinear cuBLAS
   bf16→fp32 epilogue tier — fewer exact ties at top-k, matches the bf16
   reference which casts logits to fp32). Shipped OFF: per-stream MTP hidden
   norm (A/B: spec-decode acceptance 48.4% vs 51.1% flat — checkpoint's draft
   calibrated with flat norm). Not ported (already active): padding rows are
   routed to no expert by our fork's VLLM_MOE_SKIP_PADDING=1 default
   (envs.py:215) via FusedTopKRouter -> ops.topk_* (is_padding=...), the
   is_padding mask filled in v1/worker/gpu/model_runner.py. Deferred:
   deterministic moe_align (their align.py targets the Triton fused-experts
   stack; our Marlin MoE computes token rows independently, so intra-expert
   rank order is numerically inert).

Backups next to sources: *.bak-debug (pre-patch), ngram_embedding.py.bak-fp8.

## KV corruption bug — tiering disabled (2026-10-23, late)

SYMPTOM: intermittent token-soup answers ("sm exit asset 资产..."), first generated
token, any client. Deterministic per prompt (5/5 garbage on hermes
system+tools+multi-turn at reasoning_effort=medium), nondeterministic across runs
(different soup at temp=0). Single-turn soaks stayed clean.

ROOT CAUSE: the KV offloading connector (OffloadingConnector/TieringOffloadingSpec:
8 GiB CPU tier in /dev/shm + fs tier on kv-disk) returns wrong/corrupt blocks for
cached multi-chunk prefixes. Fork-only code (v1/kv_offload/tiering/* bind-mounted
from the fork; upstream vLLM has no such tier). Not the AWQ-g32 checkpoint, not the
chat template, not reasoning_effort (medium merely picks the prefix hash whose
blocks got corrupted). GPU-only prefix cache is unaffected.

EVIDENCE: same repro prompt — 5/5 garbage with tiering, 6/6 clean after removing
kv-transfer-config; then 0/32 garbage in an 8-way concurrent battery
(medium+tools+tool-call history) with tiering off. Decode speed unchanged
(N=16 per-req 45.0 tok/s vs 44.9 with tiering).

CURRENT STATE: kv-transfer-config removed from configs/awq-1m-fp8-ssmbf16.yml
(the tiering block's exact text is preserved above). Lost: persistent warm
prefix cache across restarts + CPU hot tier. Kept: GPU prefix cache, 4.57M-token
KV pool, all speeds. VLLM_PLE_SYNC=1 kept (harmless, closes the PLE H2D race).

TO RE-ENABLE TIERING LATER: re-add the kv-transfer-config block to the config
(the exact YAML is quoted above) AND fix the bug first — candidate defects:
lookup-HIT vs async-load vs eviction race in
fs/manager.py (_account_stored/_touch_loaded), chunk offset math in submit_store,
or the spec-decode draft-group exclusion in tiering/spec.py.

## QSA top-k tie fix — ported from Morrowmake/vllm-cmp170hx (2026-10-24)

DEFECT: the stock top-k (`persistent_topk` / `cooperative_topk`) places entries
tied with the k-th score by ATOMIC ARRIVAL ORDER, so on exact score ties the
selected block SET is a function of batch shape and timing, not of the scores.
Reproduced on our GPUs (`patches/qsa-tiefix-gpu-test.py`): 8/8 rows selected
different block sets across 30 identical calls on tie-heavy scores. In serving
this silently changes which context blocks the model attends to.

FIX (upstream author: Morrowmake; file copied verbatim, md5
fe73df4449a0778904a77b6b3c878d64): `ops/qsa_topk_tiefix.py` — one Triton pass
per row that repairs only contested boundaries (tied entries go to the lowest
block indices), optionally sorting each row ascending (-1 last) in the same
launch. Integration: env gate + hooks after the decode and prefill-chunk
`_topk` in `ops/qsa_indexer.py` (fork commit 2d32a63922). K=512 absolute for
this model (indexer_budget 2048 / compress_ratio 4); reads env once at import,
before CUDA graph capture; cudagraph-safe (no host reads).

KNOBS (run-qwen38fn.sh): `QSA_CANONICAL=1` (tie repair), `QSA_SORTED=1`
(sorted rows, needs CANONICAL). Rollback: `QSA_CANONICAL=0 QSA_SORTED=0
./switch-to-awq.sh` — no rebuild.

VALIDATION (2026-10-24, CT101): GPU unit test vs independent stable-sort
oracle ALL PASS (canonical set, sorted rows, byte-identity of uncontested
rows, idempotence, 32768-col rows, empty/short/full windows). Speed A/B
same-day: ON median 113.0 vs OFF 106.3 tok/s single (sorted indices improve
KV read locality — no regression); N=16 per-req 45.2 vs baseline 45.0.
Residual temp=0 nondeterminism (coherent rewordings only) remains and is NOT
this defect: other sources (MoE alignment, MTP verify, batch-shape FP
accumulation) — Morrowmake's determinism family for those lives in their
fork's kernels and is not ported.

## 5. DFlash 2 lane (2026-10-31, patches/qwen4exp-dflash2-target.diff)

TARGET-SIDE PORT (fork e89724a801): EagleModelMixin on Qwen4ExpModel with
supports_aux_hidden_states_over_pp=True; capture = completed HC state entering
layers 6/16/26/36/46 (prev-layer pending combine materialised out-of-place via
that layer's mlp_hyper_connection, then hc-mean) — the input inco.ai trained
the drafter on (their target_layer_ids [5,15,25,35,45] + vLLM's +1). PP>1:
local aux packed into IntermediateTensors (aux_hidden_states_<slot>), mid ranks
relay upstream slots (eagle3_utils aux_hidden_state_relay_keys), last rank
collects remote + returns (sample_hidden_states, aux list). Boundary case
(pipeline cut exactly at an aux layer) captured by the owning rank; a
non-first rank never re-captures its start_layer. SupportsEagle3 on both For*
classes; models/config.py guard accepts dflash.

BOOT MATURITY LADDER (what it took, in order):
1. assert page_size_padded >= page_size in MambaSpec — GDN state page scales
   with 1+num_speculative_tokens (8 slots vs 4): block-size 1728 -> 3456
   (retention 17280 stays a multiple: 5x3456).
2. PP2 (drafter rank) fell to cudagraph_mode=NONE: auto-picked FlashInfer
   reports UNIFORM_SINGLE_TOKEN_DECODE, FULL graphs + spec-decode need
   UNIFORM_BATCH -> speculative-config attention_backend: FLASH_ATTN.
3. FLASH_ATTN rejects fp8 KV on SM80 -> speculative-config kv_cache_dtype:
   auto (draft-only bf16 KV; target keeps fp8_e4m3). ElectroHobo dodge the
   same wall by running the whole target on bf16 KV; we keep the fp8 pool.

A/B (2026-10-31, single stream, temp 0, same-day, medians): MTP-3 124.1
(prose-style). DFlash k-sched 7/5/3: prose 129.4 (acc 31%, 3.17 tok/step);
py-code 145.9 (41%); rust 189.0 (59%); code-edit 220.7 (63%, 5.40 tok/step).
Same-day MTP numbers on code were ~110-124 (older batteries 106-113). Output
distribution is lossless by construction (target verifies; drafter cannot
change answers, only speed). Rollback: SPEC_MODE=mtp ./run-qwen38fn.sh.
Draft KV adds a 5-layer bf16 group (block 64 in the group-size log line).
Pool at block 3456: 1,371,442 tokens, 1.31x of 1M — same ballpark as MTP lane.

## 6. DFlash 2 PARKED (2026-10-31 night) — rollback to MTP, work on dflash2 branches

User decision: pool >= ~5M is worth more than the TG win tonight. Production =
MTP again (kit 4fb257f + api-key commit 7d030c8; fork q38fn/stack-local @
8bad2f8c86 + working-tree kit deltas). Everything dflash lives on:
- fork branch `dflash2` @ 96556c2d0e: target-side capture (e89724a801),
  exception-safe per-group pool-charge diag in kv_cache_utils, snapshot of
  all working-tree kit deltas.
- kit branch `dflash2`: lane yml (k7 block-3456 AND k3 block-1728 variants),
  SPEC_MODE launcher knob, diag mount.
RESUME POINT: the draft's 5 sliding layers (window 2048) enter the pool as
MLAAttentionSpec with 607 blocks/request (~14 GB) instead of a windowed
SlidingWindowSpec charge (~0.5 GB honest window math: 2048-1+extra+3x6912
in-flight, ~15 blocks). Find where the SWA spec is replaced (customize_spec?
group unify? the non-causal flag?) and the pool returns to ~5M with dflash on.
CORRECTION to the earlier note: the "prefill -38%" claim was a methodology
artifact (TTFT-approx probe vs the tuning notes' dedicated prefill probe);
the same TTFT probe reads ~2.62k on BOTH the MTP and dflash lanes at 32k —
no dflash-specific prefill regression measured.
Also noted: speculative max_model_len cap (32768) does NOT change the charge.

## 7. Prefill probe correction (2026-11-01): no regression — probe bug

The "2.62k tok/s at 32k" readings (both lanes) divided an ESTIMATED 32k-token
filler by TTFT; the filler actually tokenizes to 54,435 tokens. Exact numbers
with usage.prompt_tokens, MTP production lane: 13,322 tok -> 3.91-3.98k;
54,435 tok -> 4.44-4.45k; 110,627 tok -> 4.51k tok/s — at or above the
documented 4.24-4.29k band. Same-fillers dflash lane read the same TTFTs
(~12.2s at 54k), so dflash had no prefill regression either. scripts/
prefill_exact.py in this repo is the corrected probe (use prompt_tokens,
never the char estimate).
