"""Microbench: skinny GEMM (vllm#64636 port) vs cuBLASLt on CMP 170HX (SM80).

Phase A: replicated shapes with the PR's own A100 configs.
Phase B: light config search for our wide TP=1 shapes.
Retention rule (author's): keep only >=3% faster in BOTH hot and L2-flush.
"""
import itertools
import statistics
import sys
import torch

torch.manual_seed(0)
DEV = "cuda"

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

assert shape_dynamic_skinny_gemm.is_available(), "cutlass/cute not available"
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), flush=True)

FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV)


def bench_hot(fn, iters=200, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def bench_flush(fn, iters=25, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        FLUSH.zero_()
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return statistics.median(ts)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def valid(n, k, cfg):
    if cfg.block_size % 32:
        return False
    if n % cfg.outputs_per_block:
        return False
    if k % cfg.vector_width:
        return False
    if cfg.static_k is not None and cfg.static_k != k:
        return False
    return True


def try_one(n, k, m, cfg):
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV)
    if not valid(n, k, cfg):
        return None
    try:
        out = shape_dynamic_skinny_gemm(x, w, cfg)
        ref = torch.nn.functional.linear(x, w)
        cos = cosine(out, ref)
        if cos <= 0.999:
            return ("badnum", cos)
        base_h = bench_hot(lambda: torch.nn.functional.linear(x, w))
        sk_h = bench_hot(lambda: shape_dynamic_skinny_gemm(x, w, cfg))
        base_f = bench_flush(lambda: torch.nn.functional.linear(x, w))
        sk_f = bench_flush(lambda: shape_dynamic_skinny_gemm(x, w, cfg))
        return (base_h, sk_h, base_f, sk_f, cos)
    except Exception as ex:
        return ("err", str(ex)[:60])


def report(tag, n, k, m, cfg, r):
    if r is None:
        return None
    if r[0] in ("err", "badnum"):
        print(f"{tag} ({n},{k}) M={m} {cfg} -> {r[0]}: {r[1]}", flush=True)
        return None
    base_h, sk_h, base_f, sk_f, cos = r
    up_h, up_f = base_h / sk_h, base_f / sk_f
    verdict = "KEEP" if up_h >= 1.03 and up_f >= 1.03 else "drop"
    print(
        f"{tag} ({n},{k}) M={m} bs={cfg.block_size} ob={cfg.outputs_per_block} "
        f"ku={cfg.k_unroll} vw={cfg.vector_width} sk={cfg.static_k is not None} | "
        f"base {base_h:7.2f}/{base_f:7.2f}us skinny {sk_h:7.2f}/{sk_f:7.2f}us | "
        f"x{up_h:4.2f}/x{up_f:4.2f} cos {cos:.5f} {verdict}",
        flush=True,
    )
    return (up_h, up_f, verdict)


# ---------------- Phase A: replicated shapes, PR configs ----------------
PR_CONFIGS = {
    (512, 2560): {
        1: SkinnyGemmConfig(1, 160, 2, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 160, 1, static_k=2560),
        4: SkinnyGemmConfig(4, 160, 2, k_unroll=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4, static_k=2560),
    },
    (1, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=2, static_k=2560),
        2: SkinnyGemmConfig(2, 160, 1, k_unroll=2, static_k=2560),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=4, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
    },
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, static_k=10240),
        2: SkinnyGemmConfig(2, 160, 4, k_unroll=4),
    },
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 4, k_unroll=4, vector_width=2, static_k=320),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=4, vector_width=2, static_k=320),
        4: SkinnyGemmConfig(4, 32, 4, k_unroll=4, vector_width=2, static_k=320),
    },
}

print("\n===== PHASE A: replicated shapes, PR configs =====", flush=True)
for (n, k), plans in PR_CONFIGS.items():
    for m, cfg in plans.items():
        report("A", n, k, m, cfg, try_one(n, k, m, cfg))

# ---------------- Phase B: TP1 shapes, light search ----------------
TP1_SHAPES = [
    (16384, 2560, "gdn-qkvz"),
    (21504, 2560, "qsa-qkv-gate"),
    (2560, 6144, "out-proj"),
    (1280, 2560, "shared-gateup"),
    (96, 2560, "gdn-ba"),
    (248320, 2560, "lm-head"),
]
MS = [1, 5, 10]

print("\n===== PHASE B: TP1 shapes, search =====", flush=True)
for n, k, tag in TP1_SHAPES:
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV)
    for m in MS:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
        base_h = bench_hot(lambda: torch.nn.functional.linear(x, w))
        cands = []
        for bs in ((160, 256) if n >= 100000 else (128, 160)):
            for ob in (1, 2, 4):
                for ku in (2, 4):
                    cfg = SkinnyGemmConfig(m, bs, ob, k_unroll=ku, vector_width=4, static_k=k)
                    if not valid(n, k, cfg):
                        continue
                    try:
                        shape_dynamic_skinny_gemm(x, w, cfg)
                        sk = bench_hot(lambda: shape_dynamic_skinny_gemm(x, w, cfg), iters=60, warmup=10)
                    except Exception:
                        continue
                    cands.append((sk, cfg))
        if not cands:
            print(f"B {tag} ({n},{k}) M={m}: no valid configs", flush=True)
            continue
        cands.sort(key=lambda t: t[0])
        top = [cfg for _, cfg in cands[:2]]
        # also try the fastest with vw=2 and dynamic_k variants
        sk0, cfg0 = cands[0]
        for extra in (
            SkinnyGemmConfig(m, cfg0.block_size, cfg0.outputs_per_block, k_unroll=cfg0.k_unroll, vector_width=2, static_k=k),
            SkinnyGemmConfig(m, cfg0.block_size, cfg0.outputs_per_block, k_unroll=cfg0.k_unroll, vector_width=4, static_k=None),
        ):
            if valid(n, k, extra):
                try:
                    shape_dynamic_skinny_gemm(x, w, extra)
                    sk = bench_hot(lambda: shape_dynamic_skinny_gemm(x, w, extra), iters=60, warmup=10)
                    cands.append((sk, extra))
                except Exception:
                    pass
        cands.sort(key=lambda t: t[0])
        top = [cfg for _, cfg in cands[:2]]
        print(f"B {tag} ({n},{k}) M={m} base_hot={base_h:.2f}us", flush=True)
        for cfg in top:
            report(f"B-{tag}", n, k, m, cfg, try_one(n, k, m, cfg))
    del w
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
