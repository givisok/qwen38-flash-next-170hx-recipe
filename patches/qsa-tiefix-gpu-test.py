"""GPU validation of the ported QSA top-k tie fix (canonical set + sort).

Run inside the serving container:
    mkdir -p /tmp/tiefix_test && cp qsa_topk_tiefix.py qsa_tiefix_gpu_test.py /tmp/tiefix_test/
    cd /tmp/tiefix_test && python3 qsa_tiefix_gpu_test.py

Part 1  defect demo: the stock persistent_topk on tie-heavy scores can select
        different block sets across repeated calls on THIS hardware.
Part 2  tiefix(sort=True) after every call: the set always equals the
        independent stable-sort reference and each row is sorted (-1 last).
Part 3  uncontested rows stay byte-identical without sort; idempotence; wide
        decode-like rows (32768 cols); short/empty windows.
"""
import sys

import torch

sys.path.insert(0, "/tmp/tiefix_test")
from qsa_topk_tiefix import topk_tiefix_  # noqa: E402

import vllm  # noqa: E402,F401  (loads torch.ops._C)

K = 512
DEV = "cuda"
failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        failures.append(name)


def canonical_topk(logits_cpu, k, ends_cpu):
    """Independent stable-sort oracle: (score desc, column asc), -1 fill."""
    scores = logits_cpu.tolist()
    ends = ends_cpu.tolist()
    rows, cols = logits_cpu.shape
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        window = list(range(0, min(cols, max(0, ends[r]))))
        picked = sorted(window, key=lambda c: (-scores[r][c], c))[:k]
        for s, c in enumerate(picked):
            out[r, s] = c
    return out


def row_sets(t):
    return [frozenset(v for v in r.tolist() if v >= 0) for r in t]


def sorted_ok(row):
    vals = sorted(v for v in row if v >= 0)
    return list(row[: len(vals)]) == vals and all(v == -1 for v in row[len(vals):])


def tied_logits(rows, cols, levels, gen):
    return (torch.randint(0, levels, (rows, cols), generator=gen).float() - 3) / 4


_ws = None
_ends_dtype = torch.int32


def stock_topk(logits, ends, out):
    global _ws, _ends_dtype
    if _ws is None:
        _ws = torch.empty((1 << 20,), dtype=torch.uint8, device=logits.device)
    e = ends.to(_ends_dtype)
    torch.ops._C.persistent_topk(logits, e, out, _ws, K, logits.shape[1])


def run_part1():
    global _ends_dtype
    print("== part 1: stock persistent_topk on tie-heavy scores (defect demo) ==")
    gen = torch.Generator().manual_seed(123)
    rows, cols = 8, 4096
    logits = tied_logits(rows, cols, 5, gen).to(DEV)
    ends = torch.full((rows,), cols, dtype=_ends_dtype, device=DEV)
    variants = [set() for _ in range(rows)]
    try:
        for _ in range(30):
            out = torch.empty((rows, K), dtype=torch.int32, device=DEV)
            stock_topk(logits, ends, out)
            for r in range(rows):
                variants[r].add(frozenset(v for v in out[r].tolist() if v >= 0))
    except Exception:  # dtype mismatch on the op -> retry with int64
        _ends_dtype = torch.int64
        variants = [set() for _ in range(rows)]
        for _ in range(30):
            out = torch.empty((rows, K), dtype=torch.int32, device=DEV)
            stock_topk(logits, ends, out)
            for r in range(rows):
                variants[r].add(frozenset(v for v in out[r].tolist() if v >= 0))
    variable = sum(1 for v in variants if len(v) > 1)
    print(f"  rows whose selected set varied across 30 identical calls: {variable}/{rows}")
    print("  (0 in this isolated loop does not prove serving determinism —")
    print("   atomics order under real load differs; parts 2-3 are the contract)")
    return logits, ends


def run_part2(logits, ends):
    print("== part 2: tiefix(sort=True) -> canonical set + sorted rows ==")
    ref = canonical_topk(logits.cpu(), K, ends.cpu())
    ref_sets = row_sets(ref)
    ok_set, ok_sorted = True, True
    for it in range(30):
        out = torch.empty((logits.shape[0], K), dtype=torch.int32, device=DEV)
        stock_topk(logits, ends, out)
        topk_tiefix_(logits, out, row_ends=ends, sort=True)
        cpu = out.cpu()
        if row_sets(cpu) != ref_sets:
            ok_set = False
            print(f"  iter {it}: SET mismatch vs reference")
            break
        if not all(sorted_ok(r) for r in cpu.tolist()):
            ok_sorted = False
            print(f"  iter {it}: row not sorted / -1 tail broken")
            break
    check("set equals canonical reference on every call", ok_set)
    check("rows sorted ascending, -1 tail preserved", ok_sorted)


def run_part3():
    print("== part 3: edges ==")
    gen = torch.Generator().manual_seed(999)

    # 3a: uncontested rows byte-identical without sort
    rows, cols = 4, 2048
    logits = torch.rand((rows, cols), generator=gen).to(DEV)
    ends = torch.full((rows,), cols, dtype=_ends_dtype, device=DEV)
    a = torch.empty((rows, K), dtype=torch.int32, device=DEV)
    stock_topk(logits, ends, a)
    b = a.clone()
    topk_tiefix_(logits, b, row_ends=ends, sort=False)
    check("uncontested rows unchanged byte-for-byte (sort off)", torch.equal(a, b))

    # 3b: idempotence with sort
    c = a.clone()
    topk_tiefix_(logits, c, row_ends=ends, sort=True)
    d = c.clone()
    topk_tiefix_(logits, d, row_ends=ends, sort=True)
    check("idempotent under re-application", torch.equal(c, d))

    # 3c: wide decode-like rows
    rows, cols = 2, 32768
    logits_w = tied_logits(rows, cols, 5, gen).to(DEV)
    ends_w = torch.tensor([cols, 4096], dtype=_ends_dtype, device=DEV)
    out = torch.empty((rows, K), dtype=torch.int32, device=DEV)
    stock_topk(logits_w, ends_w, out)
    topk_tiefix_(logits_w, out, row_ends=ends_w, sort=True)
    ref_w = canonical_topk(logits_w.cpu(), K, ends_w.cpu())
    cpu = out.cpu()
    check(
        "wide rows (32768/4096 cols): canonical set",
        row_sets(cpu) == row_sets(ref_w),
    )
    check("wide rows: sorted with -1 tail", all(sorted_ok(r) for r in cpu.tolist()))

    # 3d: short and empty windows
    rows, cols = 4, 8192
    logits_s = tied_logits(rows, cols, 5, gen).to(DEV)
    ends_s = torch.tensor([0, 3, K, K - 1], dtype=_ends_dtype, device=DEV)
    out = torch.empty((rows, K), dtype=torch.int32, device=DEV)
    stock_topk(logits_s, ends_s, out)
    topk_tiefix_(logits_s, out, row_ends=ends_s, sort=True)
    ref_s = canonical_topk(logits_s.cpu(), K, ends_s.cpu())
    cpu = out.cpu()
    check(
        "empty/short/full windows: canonical set",
        row_sets(cpu) == row_sets(ref_s),
    )
    check("empty/short/full windows: sorted", all(sorted_ok(r) for r in cpu.tolist()))


def main():
    logits, ends = run_part1()
    run_part2(logits, ends)
    run_part3()
    print()
    if failures:
        print("RESULT: FAIL —", failures)
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
