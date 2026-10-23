# Fork patches required for leoncca AWQ-g32 checkpoint (2026-10-23)

Two patches on top of q38fn/stack (bind-mounted by run-qwen38fn.sh):

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
(backup with tiering: awq-1m-fp8-ssmbf16.yml.bak-tiering). Lost: persistent warm
prefix cache across restarts + CPU hot tier. Kept: GPU prefix cache, 4.57M-token
KV pool, all speeds. VLLM_PLE_SYNC=1 kept (harmless, closes the PLE H2D race).

TO RE-ENABLE TIERING LATER: restore .bak-tiering config block AND fix the bug
first — candidate defects: lookup-HIT vs async-load vs eviction race in
fs/manager.py (_account_stored/_touch_loaded), chunk offset math in submit_store,
or the spec-decode draft-group exclusion in tiering/spec.py.
