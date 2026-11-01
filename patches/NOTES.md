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

## 8. RESOLVED (2026-11-01): pool collapse root cause + fix — 5.24M tokens / 5.00x with dflash ON

Branches `dflash2-pool` (fork 4beee19d43, kit 0a95834), created from dflash2.

Full per-group + per-worker [pool-diag] showed the real mechanism — NOT the
suspected SWA-to-MLA substitution (the MLAAttentionSpec/607-block line is the
TARGET's own 8-layer attention group, present in the MTP lane too):

- pool num_blocks = min over PP ranks of available // bytes_per_block, where
  bytes_per_block = max over groups of the summed per-layer page of a group.
- The draft's SlidingWindowSpec was honest (window 2048, 19 blocks/req at
  block 1728) but its per-layer page = 1728 tok x 4096 B (8 KV heads x 128 x
  2 x bf16) = 7,077,888 B, so the 5-layer draft group summed to 35,389,440 B
  on PP2 (the drafter's rank) — 4x the GDN group's 8,847,360 B.
- PP2 bytes_per_block 8.85MB -> 35.4MB cut num_blocks 3492 -> 939, and the
  shared min collapsed the whole pool to 1,519,464 tokens / 1.45x.

Fix: `VLLM_SWA_BLOCK_CAP` env clamp in
model_executor/layers/attention/attention.py::get_kv_cache_spec (limits the
SWA block after _largest_kernel_block_within). Set to 432 in the kit launcher
(SWA_BLOCK_CAP knob, default 432, dflash lane): draft group page becomes
5 x 432 x 4096 = 8,847,360 B == the GDN group page, so bytes_per_block stays
at the GDN level on every rank. Draft charge rises 19 -> 70 blocks/req
(window 2048 + 3x6912 in-flight over a 432-token block) — cheap.

Resulting lane (k=3, block 1728, chunk 6912, FLASH_ATTN draft, bf16 draft KV):
- pool 5,238,379 tokens / 5.00x (prod MTP: 5,191,201 / 4.95x) — dflash now
  FREE on pool, slightly ahead of MTP.
- battery 2026-11-01: solo TG py 145.3 / rust 166.5 / edit 170.8 / sql 257.9
  tok/s; acceptance 73.9%, 3.22 tok/step; N=6 aggregate 566 tok/s (same-probe
  MTP: 587); needle 64k PASS; greedy non-identical (pre-existing).
- prefill exact: 13,385 tok -> 3.66-3.68k; 54,112-54,435 -> 4.35-4.39k;
  110,627 -> 4.41k tok/s — parity with the MTP lane.

Not promoted to production yet: N=6 aggregate still ~4% under MTP and k-sched
(7/5/3 vs flat 3) untested at the new pool. Decision pending.

## 9. Concurrency sweep N=8/16/32 (2026-11-01): all three lanes equal within
noise at load; k7 keeps the solo crown — k7 promoted to default dflash lane

Same-probe aggregate (characters/s, 32 distinct code prompts x 400 tok,
streamed; repeat on MTP showed +/-6% noise):

| lane | N=8 | N=16 | N=32 | solo TG py/rust/edit/sql | pool |
|---|---|---|---|---|---|
| MTP-3 | 2026 | 3140 (2951) | 3010 (2981) | ~124 / ~189 / ~221 / - | 5.21M |
| dflash k=3 | 2027 | 3432 | 2896 | 145/167/171/258 | 5.24M |
| dflash k=7 (7/5/3) | 2172 | 3047 | 2795 | 160/199/223/371 | 4.01M |

At saturated batch all lanes sit in the same ~2.8-3.4k band (speculation
gain dies with concurrency — schedule drops to k=3 and the verify overhead
cancels the draft). The lanes differ where load is light: k7 wins N=8 (+7%)
and crushes solo (+10..44% over k3, up to +80% over MTP on sql, 4.71 tok/step).
k3's N=16 3432 was the single best cell but within 2x-noise of MTP's band.

Decision: k7 (block 3456, chunk 10368, schedule [[1,2,7],[3,4,5],[5,16,3]],
SWA_BLOCK_CAP=432) is the default dflash lane — pool 4.01M clears the 4M bar,
solo is the point of a homelab box, and nothing is lost under load. Rollback
ladder: k3 = block 1728/chunk 6912/k3 (history 0a95834..7f1ccd0), prod MTP =
SPEC_MODE=mtp ./run-qwen38fn.sh.

## 10. Verification with the sparkDash protocol (2026-11-01): my char-based
sweep was directionally right; canonical token-based numbers CONFIRM k7

The concurrency sweep in section 9 counted streamed characters (stdlib probe).
Re-measured all three lanes with an exact replica of the sparkDash DecodeBench
protocol (MiaAI-Lab/sparkDash @ 3e6d4ef — the protocol both AlexBrandes and
ElectroHobo publish their numbers with): chat/completions, stream,
include_usage, temperature 0, top_p 1, thinking off, max_tokens 400, a
32-token warmup before each wave, decode tok/s = (completion_tokens - 1) /
(tLast - tFirst) on content-chunk timestamps, 3 passes per level, medians.
Replica: scripts/sd_bench.py (their prompt pools verbatim).

Per-request median decode tok/s (code tasks / prose / structured=1..200):

| lane | N=1 | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| MTP-3 | 169 / 150 / 206 | 103 / 89 / 140 | 79 / 65 / 102 | 66 / 58 / 94 |
| dflash k=3 | 197 / 158 / 206 | 110 / 89 / 139 | 84 / 66 / 103 | 63 / 58 / 94 |
| dflash k=7 | **296 / 166 / 360** | **134 / 93 / 146** | **81 / 68 / 74** | 69 / 52 / 79 |

Aggregate tok/s (sum of decode tokens over the concurrent first->last window):

| lane | code N=8/16/32 | prose N=8/16/32 | structured N=8/16/32 |
|---|---|---|---|
| MTP | 648 / 995 / 870 | 657 / 956 / 869 | 1023 / 1507 / 1398 |
| k=3 | 696 / 975 / 844 | 645 / 964 / 876 | 1019 / 1514 / 1430 |
| k=7 | **782 / 1043 / 860** | **687 / 983 / 763** | 1050 / 1123 / 1155 |

Verdict unchanged, now on the canonical scale: k7 wins solo massively
(code +75% over MTP, structured +75%), wins or ties code/prose aggregate
through N=16, and only the highly-predictable structured count-to-200 stream
(a near-100% draft-acceptance workload) favors k3/MTP at N>=16 — exactly the
regime where any drafter's edge dies. k7 stays the default lane.

## 11. REGRESSION FOUND (2026-11-01 evening): DFlash lanes break prefix-cache
turnaround — agent workloads (pi) are FASTER on MTP; prod reverted to MTP

User report: pi agent at 32k+ context shows 75-90 tok/s on dflash k7 vs
105-120 previously on MTP. Not reproducible by any single-stream decode probe:
k7 at 45k real tokens, temp 1.0, top_p 0.95, thinking on/off, 900-token
generations, random (non-predictable) filler: code 245-368, prose 134-155,
thinking-answer 193-210 tok/s — all >= MTP (173-206 / 127-139 / 155-164).
Decode-at-depth is FLAT 32k->128k on every lane.

The regression is in PREFIX-CACHE turnaround (scripts/cache_probe.py; 45k
random-token prompt, TTFT incl. cached_tokens from usage):

| probe | MTP (b1728) | dflash k3 (b1728) | dflash k7 (b3456) |
|---|---|---|---|
| identical resend | 0.26-0.89s, cached 43.2-45.7k | run1-2: 0 cached 10.3-11.7s, run3: 0.89s | 1.41-1.45s, cached 41.5k |
| +3k grown head, resend | 0.27s, cached 50.1k | 11.3-11.7s, cached 0 | 11.7-12.4s, cached 0 |
| +3k grown tail | 1.16-1.63s | 1.94-2.41s | 2.55-2.99s |

An agent turn = cached-prefill turnaround + decode. Per turn at 32k+ dflash
adds +1.3..+11s of re-prefill vs MTP while saving ~1s of decode — pi's
perceived tok/s (output tokens / wall per turn) collapses exactly as
reported. Not block-size-dependent (k3 b1728 shows it too): the draft's
5-layer SWA group (window 2048) participates in block hashing/eviction —
window-bounded groups free "skipped" blocks, so the full-prefix hash never
matches and repeated prompts re-prefill thousands of tokens. Root-cause trace
(draft group vs prefix cache hash/evict) is the new resume point on the
dflash2-pool branches.

DECISION: production reverted to MTP (SPEC_MODE=mtp; pool 5,206,204 / 4.97x,
health 200). prod-dflash-k7-20261101 remains the marker of the best-solo
dflash state (bench-only); prod pointer for serving = prod-mtp-20261101.
Solo decode wins stand: k7 296 code / 360 structured vs MTP 169/206 — use
dflash for one-shot short-context work until the cache interaction is fixed.

## 12. ROOT-CAUSED and FIXED (2026-11-01 night): the draft SWA group silently
disabled fine-grained prefix hits for the WHOLE engine — agent turns re-prefilled

Instrumented the hit machinery (VLLM_CACHE_DIAG: per-group [hit-diag] in the
coordinator, [cache-diag] masks in managers, [hash-ins]/[hash-rm] in the pool).
The chain, confirmed line by line in the logs:

1. The draft's SlidingWindowSpec group makes partial_hash_hits_enabled()
   return False — SlidingWindowManager has no fine-grained lookup support —
   so the boot logs "Disabling fine-grained prefix-cache hits ..." and the
   engine drops from 16-token hit granularity to whole 1728/3456 blocks.
2. With block-aligned hits, MambaManager.lookup only probes full-block
   hashes. Align-mode GDN exports its prefill states as PARTIAL/checkpoint
   hashes (chunk boundaries) and nulls most interior blocks, so almost no
   full-block mamba hash ever exists: request #1's lookup walked i=24..20
   all-miss while MLA had already hit 43200.
3. Mamba gates the reconciled min -> FINAL hit=0 -> identical resend did a
   FULL 10-12s re-prefill. Only the second identical resend hit (decode-phase
   caching had by then registered the boundary block). Agent turns (new
   prompt shape every turn) always pay the miss — pi at 32k+ showed 75-90.

FIX: VLLM_SWA_NO_PREFIX_CACHE=1 (fork 9eadcf0a8f, kit knob default 1) — the
drafter's spec becomes _NonCacheableSlidingWindowSpec: the draft SWA group
opts out of prefix caching entirely (like circular buffers). Its 2048-token
window KV is recomputed on the resumed tail for ~free, and the fine-grained
gate no longer trips for the target groups. Boot warning gone.

cache_probe after the fix (45k random-token prompt, TTFT incl. cached):
- k3 lane: identical resend 0.89s/43200 then 0.26s/45728 (was 10.3s/0);
  grown-head resend 0.26s/50112 (was 11.3s/0); grown-tail 1.18-1.70s.
- k7 lane: identical 1.44s then 0.26s; grown-head resend 1.61s/44928
  (was 11.7s/0); grown-tail resend 0.27s/50064 (was 2.5-3.0s).
Speeds and quality unchanged (k3: code 200 / prose 155 / structured 208
solo; k7: 299/181/359, N=8 aggregates 741/692/1041; needle 64k PASS both).

dflash is now plausibly strictly better than MTP: same cache turnaround,
+50-75% solo decode, equal concurrency. Verify with a real pi session before
promoting; prod pointer still prod-mtp-20261101 while it brews.

## 13. PROMOTED (2026-11-01 late): dflash k7 + cache fix is production

Live pi session validated the fix (session 2026-10-31T21-28-01, two stretches
on the same day): MTP stretch median out/s ~88 (out+reasoning ~130), k7+fix
stretch ~100 (~170, peaks 186/293), cacheRead growing 72k->123k over 28 turns.
pi's counter divides output tokens by full turn duration (prefill included) —
honest, which is exactly why the pre-fix cache storm showed up there first.

Promotion state:
- fork q38fn/stack-local @ 3c367a58e5 (merge of dflash2-pool 9eadcf0a8f;
  content identical to the branch, both lanes boot-tested from this tree)
- kit main @ f248264 (merge 92dfe8d + CACHE_DIAG set -u fix)
- serving: k7 lane (block 3456, chunk 10368, schedule 7/5/3, FLASH_ATTN
  draft, bf16 draft KV, SWA_BLOCK_CAP=432, SWA_NO_PREFIX_CACHE=1),
  pool 4,014,818 tokens / 3.83x
- prod boot re-verified from main: cache probe healthy (resend 0.26-1.50s,
  grown-head 1.59s, grown-tail 0.27s), sparkDash solo code 292 tok/s
- rollback ladder: k3 lane = block 1728/chunk 6912/k3; MTP = SPEC_MODE=mtp
  ./run-qwen38fn.sh (state prod-mtp-20261101)
Tag: prod-dflash-k7-20261101 (moved to this state).

## 14. The "why does clee get 7M" question + MTP-4 (2026-11-01)

clee/qwen3.8-flash-next-3x170hx reports "KV pool: 7.36M tokens (7.01x)".
Reproduced the regime on our box by peeling our stack (nospec probe yml in
the kit): MTP off -> 5,932,866 / 5.66x (+0.73M: that is the price of the
GDN spec slots + draft weights); + VLLM_PP_LAYER_PARTITION=16,16,16 +
mamba-ssm-cache-dtype bfloat16 -> 7,102,105 / 6.77x. So their 7M is
"no speculative decoding + even partition + bf16 mamba state" — a
throughput-of-decode tradeoff (their own numbers: ~74 tok/s single stream,
no spec) that we deliberately do not make: dflash k7 gives 292-299 code
tok/s for the pool price of 4.01M.

MTP-4 tested (kit yml awq-1m-fp8-ssmbf16-mtp4.yml, graphs on a 5-lattice to
16x(1+4)=80): boots at block 1728 (no GDN assert — slots 5 fit the fp32
page), pool 5,163,367 / 4.92x. sparkDash solo code 182.6 / prose 152.8 /
structured 238.8 tok/s (MTP-3: 169.3/149.6/205.9) = +8-16% over MTP-3;
N=8 code aggregate 672 (MTP-3 648, dflash k7 741); battery acceptance 63.9%,
3.56 tok/step, N=6 aggregate 610. MTP-4 replaces MTP-3 as the fallback lane
default (same pool class, faster). Still well under dflash k7 solo.

dflash k7 + bf16 mamba state: NO pool gain (4,012,456 / 3.83x, identical) —
with the GDN pages halved the binding group becomes the target's MLA fp8 KV
raw bytes; k7 also still needs block 3456 (GDN assert fires at 1728 even
with bf16). Reverted to fp32 mamba state (known-good, validated prod).

## 15. Fresh N=8/16/32 re-measure: dflash k7 vs MTP-4 (2026-11-01 evening)

Same-hour sparkDash sweeps, aggregate tok/s (per-request medians in parens):

| workload | lane | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| code | k7 | 673 (116) | 989 (75) | 834 (66) |
| code | MTP-4 | 693 (108) | 948 (74) | 881 (67) |
| prose | k7 | 672 (93) | 946 (64) | 753 (50) |
| prose | MTP-4 | 651 (86) | 921 (62) | 814 (54) |
| structured | k7 | 981 (133) | 1074 (71) | 1095 (74) |
| structured | MTP-4 | **1123 (154)** | **1583 (107)** | **1490 (99)** |

Verdict unchanged and now measured same-hour: on CODE (what agents emit)
the two lanes are equal within noise at every load 8..32 — per-request
medians are within 5-8 tok/s of each other everywhere. MTP-4 pulls ahead
only on predictable structured streams at load (+36..47% at N=16/32) where
its 4-draft verification is cheap and near-100% accepted. Since the pool
difference (4.01M vs 5.16M) buys 60%+ solo speed on k7 and costs nothing at
agent-style load, dflash k7 stays production. Flip to MTP-4 for batch jobs
that emit JSON/CSV at high concurrency.

## 16. Full sweep N=1/2/4/8/16/32: dflash k7 vs MTP-4 (2026-11-01 night)

Per-request median tok/s (aggregate below each):

code:
| N | k7 | MTP-4 | delta |
|---|---|---|---|
| 1 | 295 (295) | 195 (195) | +51% |
| 2 | 245 (479) | 159 (313) | +54% |
| 4 | 228 (652) | 138 (445) | +65% |
| 8 | 108 (707) | 113 (686) | ~0 |
| 16 | 78 (962) | 77 (978) | ~0 |
| 32 | 67 (838) | 67 (850) | ~0 |

prose: k7 152/151/110/93/64/52 vs MTP-4 149/129/108/85/61/55 — k7 ahead
through N=4, parity after. structured: k7 355/309/211 vs 241/214/191 at
N=1/2/4 (+32..+48%), MTP-4 ahead from N=8 (154/107/98 vs 133/93/76).

The pi-with-subagents zone (N=2-4) is where dflash k7 earns its keep:
+54..65% per-request on code vs MTP-4. At N>=8 code is a tie, so the pool
spent on the drafter costs nothing under load. Structured batch generation
remains the only MTP-4 win. k7 stays production.

## 17. PROMOTED (2026-11-01 final): production = dflash k=5 + bf16 mamba state

The "why not k4" question led here. fp32 GDN slots: k3 (4 slots) is the max
that fits the block-1728 page (k4 = 5 x 442KB = 2.21MB > 1.77MB, assert).
Halving the slot with mamba-ssm-cache-dtype bfloat16 lets k=5 (6 x 221KB =
1.33MB) fit block 1728 with margin — and lands the pool exactly in the
user's MTP-4 target band.

State: kit main @ 078b536 (dflash.yml = k5+bf16 @ block 1728, chunk 6912;
variant copy awq-1m-fp8-ssmbf16-dflash-k5bf16.yml), fork q38fn/stack-local @
3c367a58e5 (unchanged — config-only move). Serving pool 5,185,037 / 4.94x.

Measured (same-day, sparkDash / scripts):
- solo code 253-257 / prose 173 / structured 285 tok/s
  (k7: 295/152/355; k3: 197/155/206; MTP-4: 195/149/241)
- N=2/4 code per-req 213/175 (k7 245/228, MTP-4 159/138)
- N=8 aggregates code 832 / structured 1268 / prose 672 — BEST code and
  structured N=8 of every lane tested, incl. MTP-4 (686/1119)
- battery acceptance 60.6%, 4.03 tok/step; decode at 128k flat (code 269)
- bf16 GDN state quality gates: needle 64k PASS AND 120k PASS
  (scripts/needle_deep.py — secret buried mid-120k random records, exact
  answer); prefix cache at 95k: grown-head resend 0.41s / 95,584 of 95,600
  tokens cached; tail resend 0.97s
Rollback ladder: k7 lane (block 3456/fp32, tag prod-dflash-k7-20261101),
MTP-4 (SPEC_MODE=mtp + mtp4 yml), MTP-3 (prod-mtp-20261101).

## 18. Full-depth 1M multi-needle gate (2026-11-01): 18/18 PASS
Question: does retrieval actually work at the full YaRN'd 1M on the prod
k5+bf16 lane? Probe scripts/needle_1m.py: ONE 1,010,326-token wall of the
unpredictable rec/sku filler with six needles ("secret access code for
station X is NNNNNN") at 5/20/40/60/80/95 % depth, each asked 3 times with
different question phrasings, plus a negative control on an absent station
(ZULU). Doc is shared by all queries, so query 1 is the cold prefill and
queries 2-19 must ride the prefix cache.
- 18/18 PASS, exact six-digit codes, every depth, every phrasing. No
  cross-needle confusion; the ZULU control answered an invented 000000
  (format hallucination, no real code leaked).
- Cold prefill 1,010,360 tok = 235.6 s = 4,287 tok/s — the 4.24-4.51k band
  holds at the full 1M depth (YaRN x4 costs nothing measurable here).
- Cache fix at full scale: every resend cached 1,007,424 / 1,010,360 tokens
  (99.7 %), TTFT 4.3-4.5 s vs 235.6 s cold (~52x). SWA_NO_PREFIX_CACHE fix
  validated end-to-end at 1M, not just at 95k.
- Calibration correction: this filler is 1.5203 chars/token (measured via
  /tokenize), NOT the 1.763 the old probes assumed. Consequences: the first
  1M attempt 400-rejected (real size 1,160,006 > max_model_len 1,048,576);
  and needle_deep.py's nominal "120k" run was actually ~139k tokens (its
  PASS claim only gets stronger). needle_1m.py self-calibrates: build ->
  /tokenize -> rescale -> /tokenize (converged in one iteration, +326 tok).

## 19. DFlash speculation dies at 262,144 = YaRN boundary (2026-11-01 night)

User saw pi at ~40 tok/s. Server-side truth: pi's counter was honest - the
engine itself logged 30-46 tok/s generation for a SINGLE running request.
The active session (running since Oct 31) crossed 262,144 tokens at
20:12Z and per-turn tok/s stepped from 84-147 down to 38-42 within a
minute and stayed there. Not the cache (hit 93.5%, cacheRead keeping up,
only ~1k new tokens per turn), not preemption (0), not thermals
(35-38 C, full clocks, no throttle reasons), not other clients.
Live spec counters during the slow window: +707 drafts, +18 accepted =
0.025 tokens/draft, vs a healthy 1.81/draft average since boot.
Paper cause: the DFlash2 drafter config has max_position_embeddings
262144 and rope_scaling: NONE - it has no YaRN. The target runs
yarn x4 from original 262144. Past 262,144 the two models compute in
incompatible positional systems; every draft is rejected; with k=5 each
step still pays verification of 5 draft positions through the full
context (~6x step cost for ~1 emitted token).
Controlled probe (scripts/spec_gate_depth.py, self-calibrated filler,
temp 1, 200 tok, spec counter deltas around each request):
- @250,080 tok: 72 drafts / 128 accepted = 1.78 acc/draft, 127.4 tok/s
- @270,109 tok: 199 drafts / 0 accepted = 0.00 acc/draft, 46.2 tok/s
Exactly reproduces the pi symptom. Target quality is NOT affected -
the 1M multi-needle gate (18) passed at 1.01M: retrieval works, only
speculation is dead weight past the boundary.
Mitigations: compact agent sessions below ~250k (immediate fix); a fork
gate that skips drafting past 262144 would give plain-decode floor
(~90-110 tok/s at that depth) instead of 46; MTP lane untested >262k
(MTP head is part of the target network and shares its rope, so it
plausibly survives - needs the same probe on the MTP lane before
promising).

## 20. Full depth sweep 32k-1M (2026-11-01 night): flat until the boundary, cliff after

spec_sweep_depth.py: rec-filler self-calibrated via /tokenize, prose task,
temp 1 / top_p 0.95, 200-token decode, spec-counter deltas, quiet-check
(waits for 0 running requests so foreign traffic cannot pollute counters).
Prose/temp-1 protocol - absolute numbers are lower than sparkDash temp-0
medians; the CURVE is the point.

| depth (tok) | prefill tok/s | decode tok/s | acc/draft |
|---|---|---|---|
| 32,041 | 4,238 | 127.0 | 1.80 |
| 64,023 | 4,360 | 108.7 | 1.40 |
| 128,073 | 4,410 | 133.2 | 1.91 |
| 256,145 | 4,421 | 132.4 | 1.87 |
| 512,253 | 4,388 | 46.8 | 0.01 |
| 1,000,566 | 4,294 | 47.2 | 0.01 |

Readings:
- Prefill is FLAT 4.24-4.41k tok/s from 32k all the way to 1M - no depth
  penalty at all.
- Below 262,144 decode and acceptance are flat too (109-133 tok/s,
  1.4-1.9 acc/draft; the 64k dip is one 200-token temp-1 sample, content
  noise, not a depth effect).
- Above the boundary acceptance is 0.01 (1 of 198 drafts at both depths)
  and decode is PINNED at ~47 tok/s - identical at 512k and 1M. The
  penalty is the constant k=5 verify batch with zero accepted drafts, not
  a context-length cost: the GDN layers are O(1)-state and only MLA reads
  KV, so the dead-spec penalty does not grow with depth.
- First decode measurement at 1M: 47.2 tok/s (with dead drafter). Plain
  no-spec floor at depth is unmeasured (needs a lane change); a fork gate
  that skips drafting past 262,144 would recover part of the ~2.8x cliff,
  exact size TBD on the nospec lane.

## 21. Production switched to MTP-4 (2026-11-01 night) — sweep confirms no cliff

User decision after the boundary discovery (19-20): switch to MTP. Lane:
SPEC_MODE=mtp + configs/awq-1m-fp8-ssmbf16-mtp4.yml (k=4, bf16 mamba
state, standalone mtp-fp8 draft). Pool 5,163,367 / 4.92x. Boot ~260 s.
Same sweep (spec_sweep_depth.py, prose, temp 1, quiet-check):

| depth (tok) | prefill tok/s | decode tok/s | acc/draft |
|---|---|---|---|
| 32,041 | 4,242 | 121.4 | 1.68 |
| 64,023 | 4,443 | 119.9 | 1.64 |
| 128,073 | 4,564 | 128.9 | 1.83 |
| 256,145 | 4,597 | 135.2 | 1.90 |
| 512,253 | 4,541 | 114.7 | 1.40 |
| 1,000,566 | 4,469 | 134.1 | 1.63 |

No cliff anywhere: the MTP head shares the target's rope, acceptance
stays 1.4-1.9 through 1M, decode 115-135 tok/s flat (dflash k5 at the
same depths: 46.8/47.2 with acc 0.01). Prefill is slightly FASTER than
the dflash lane deep (4.5-4.6k vs 4.3-4.4k - the drafter's five layers
taxed prefill ~3%). Below-boundary prose is parity (121-135 vs 109-133,
same protocol); dflash keeps the solo-code crown (sparkDash 253-257 vs
195) for sessions that stay under ~250k. Trade accepted: ~20% slower
solo code below 250k in exchange for 2.8x decode at 512k-1M and no
compaction cliff for long agent sessions. Rollback: SPEC_MODE=dflash
(launcher default) or MTP-3 via the base yml.
