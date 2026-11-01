# Benchmark and probe scripts

All run from CT 101 against the live server (stdlib only, no deps).

| Script | What it measures |
|---|---|
| `sd_bench.py` | sparkDash DecodeBench protocol replica (temp 0/top_p 1, thinking off, warmup 32, decode tok/s = (completion_tokens-1)/(tLast-tFirst), 3 passes, medians). The canonical cross-lane benchmark; args = concurrency levels. |
| `std_battery.py` | Standard gate battery: solo TG on code tasks, N=6 aggregate, cold prefill TTFT, needle 64k, greedy reproducibility. |
| `cache_probe.py` | Prefix-cache turnaround: identical resend, +3k grown head, +3k grown tail (TTFT + cached_tokens from usage). THE probe that caught the draft-SWA cache bug. |
| `needle_deep.py` | Needle retrieval at 120k: secret buried mid-wall of random records, exact-answer check (bf16 mamba state gate). |
| `needle_1m.py` | Full-depth gate: ~1.01M-token wall (self-calibrated via `/tokenize`), six needles at 5–95 % depth × 3 question phrasings + negative control; reports PASS/leak, cached_tokens, TTFT. First query is the cold prefill, the rest must ride the prefix cache. |
| `depth_probe_code.py` | Decode at real depth (calibrated filler 0.567 tok/char): code+prose at 32k/64k/128k. |
| `depth_probe_rand.py` | pi-like probe: temp 1.0/top_p 0.95, random (non-predictable) filler, 900-token generations, windowed speeds. |
| `pi_like_probe.py` | Same with thinking variants; replays how a coding agent actually hammers the API. |
| `conc_sweep.py` | Crude concurrency sweep (character-counting; superseded by sd_bench.py — kept for history). |
| `prefill_exact.py` | Exact cold prefill tok/s from usage.prompt_tokens (never divide an estimated token count by TTFT). |
| `bench_draft_ab.py`, `bench_conc.py`, `probe_contention.py`, `ab_measure.py`, `bench_ple_ckpt.py` | Tuning-series probes (see TUNING-2026-10-27). |
| `prefill.py`, `verify16.sh`, `oomtest.py` | Bring-up era probes. |
| `run-qwen38fn.sh` | Kit launcher copy (lane selection, knobs). |
| `quantize_lm_head.py` | W4A16 draft lm_head quantizer (AlexBrandes port). |
