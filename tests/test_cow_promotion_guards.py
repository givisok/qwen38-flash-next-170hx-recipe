# Guards for the partial-hash machinery: reproduction tests for the two
# 2026-11-03/04 engine crashes plus the consumer-side sibling.
import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test
init_none_hash(sha256)

H = 2  # hash granularity (fine-grained prefix match unit)
B = 6  # cache block size (multiple of H)
GID = 0


def make_request(request_id, prompt_token_ids, hash_block_size, hash_fn):
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, hash_fn),
        session_id=None,
    )


def boundary_hash(req, num_tokens):
    return req.block_hashes[num_tokens // H - 1]


def _pool(n=8):
    return BlockPool(
        num_gpu_blocks=n, enable_caching=True, hash_block_size=H
    )


# ----------------------------------------------------------------- pool level

def test_promotion_true_partial_still_promotes():
    """Regression: a genuine partial publication promotes to the full hash."""
    req = make_request("r", list(range(12)), H, sha256)
    pool = _pool()
    blocks = pool.get_new_blocks(2)
    pool.cache_partial_block(
        request=req, block=blocks[1], num_tokens=10,
        kv_cache_group_id=GID, block_size=B,
    )
    assert blocks[1].block_hash_num_tokens == 10
    pool.cache_full_blocks(
        request=req, blocks=blocks, num_cached_blocks=1,
        num_full_blocks=2, block_size=B, kv_cache_group_id=GID,
    )
    assert blocks[1].block_hash_num_tokens == 12
    h10 = make_block_hash_with_group_id(boundary_hash(req, 10), GID)
    assert not pool.cached_block_hash_to_block.contain(h10, blocks[1].block_id)
    h12 = make_block_hash_with_group_id(boundary_hash(req, 12), GID)
    assert pool.cached_block_hash_to_block.contain(h12, blocks[1].block_id)


def test_promotion_aligned_publication_is_noop():
    """Crash 2026-11-04 14:58 UTC: an aligned (block-multiple) landing or
    checkpoint publication registers the chain hash AT the full token count;
    the promotion pass used to assert count < full and kill the engine.
    Now it must be a no-op with the entry intact."""
    req = make_request("r", list(range(12)), H, sha256)
    pool = _pool()
    blocks = pool.get_new_blocks(2)
    pool.cache_partial_block(
        request=req, block=blocks[1], num_tokens=12,  # 12 % B == 0: aligned
        kv_cache_group_id=GID, block_size=B,
    )
    assert blocks[1].block_hash_num_tokens == 12
    pool.cache_full_blocks(
        request=req, blocks=blocks, num_cached_blocks=1,
        num_full_blocks=2, block_size=B, kv_cache_group_id=GID,
    )
    assert blocks[1].block_hash_num_tokens == 12
    h12 = make_block_hash_with_group_id(boundary_hash(req, 12), GID)
    assert pool.cached_block_hash_to_block.contain(h12, blocks[1].block_id)


# -------------------------------------------------------------- manager level

def _manager(num_blocks=24):
    from math import lcm

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=H, num_kv_heads=1, head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=4, shapes=(1, 1), dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        cfg,
        max_model_len=512,
        enable_caching=True,
        hash_block_size=H,
        scheduler_block_size=lcm(H, 4),
    )
    mamba_mgrs = [
        m for m in manager.coordinator.single_type_managers
        if type(m).__name__ == "MambaManager"
    ]
    assert len(mamba_mgrs) == 1
    return manager, mamba_mgrs[0]


def _complete(manager, req, prompt_len):
    """Admit, fully allocate, run the completion caching pass."""
    manager.get_computed_blocks(req)
    manager.allocate_slots(req, prompt_len, 0)
    manager.cache_blocks(req, prompt_len)


def test_producer_stale_record_degrades():
    """Crash 2026-11-03 18:22 UTC: the producer-side landing record went
    stale between publication and the next allocation (the block table
    moved). Old code: AssertionError in allocate_new_blocks. Now: warning +
    retraction, request continues."""
    manager, mamba = _manager()
    prompt = list(range(15))  # unaligned end: landing published at 14
    req = make_request("p", prompt, H, sha256)
    _complete(manager, req, 15)

    rec = mamba._partial_hit_reqs.get(req.request_id)
    assert rec is not None, "completion must record the landing CoW"
    block_idx, source_block = rec

    # Simulate a mid-table rewrite to a DIFFERENT block (the class of
    # rewrite _apply_cow performs): content identity is broken, the record
    # is stale and the unproven publication must be retracted.
    blocks = mamba.req_to_blocks[req.request_id]
    other = mamba.block_pool.get_new_blocks(1)[0]
    blocks[block_idx] = other

    manager.allocate_slots(req, 1, 15)  # next (decode) step
    # survived; the STALE record is consumed (a NEW landing record on
    # `other` is legitimate tail bookkeeping, not the stale one); the
    # unproven publication is retracted
    rec2 = mamba._partial_hit_reqs.get(req.request_id)
    assert rec2 is None or rec2[1] is not source_block
    assert source_block.block_hash is None


def test_consumer_partial_hit_cow_happy_path():
    """Consumer-side CoW regression: a mid-block partial hit lands on the
    producer's 14-token state; the first allocation must redirect the shared
    tail block to a private CoW copy and clear the record. (The record is
    created and consumed inside the same allocate_slots call, so unlike the
    producer side it cannot go stale across steps.)"""
    manager, mamba = _manager()
    producer = make_request("0", list(range(15)), H, sha256)
    _complete(manager, producer, 15)
    manager.free(producer)
    manager.new_step_starts()

    consumer_prompt = list(range(14)) + [100, 101, 102, 103, 104, 105]
    consumer = make_request("1", consumer_prompt, H, sha256)
    manager.get_computed_blocks(consumer)
    # the record appears lazily inside allocate_slots, not at lookup time
    assert mamba._partial_hit_reqs.get(consumer.request_id) is None

    manager.allocate_slots(consumer, 6, 14)
    # The consumer-side record was consumed; what remains is the consumer's
    # OWN landing publication awaiting the next allocation. Note 20 %
    # mamba_block == 0: an ALIGNED publication — the exact crash-2026-11-04
    # precondition, produced here by the completely ordinary flow. The mamba
    # table keeps only the live tail state (leading entries are null).
    rec = mamba._partial_hit_reqs.get(consumer.request_id)
    assert rec is not None, "completion must leave the landing record"
    assert rec[0] == 4 and rec[1].block_hash_num_tokens == 20
    blocks = mamba.req_to_blocks[consumer.request_id]
    assert blocks[4] is rec[1]
    assert not blocks[4].is_null and blocks[4].ref_cnt >= 1


def test_producer_slot_vacated_by_relocation_keeps_publication():
    """Prod 2026-11-05 (40 events): _relocate_speculative_block moved the
    published tail block to the table end and nulled its old slot. The
    publication still describes the request's true history — it must be
    KEPT (only the directed CoW shortcut is lost), not retracted."""
    manager, mamba = _manager()
    prompt = list(range(15))  # unaligned end: landing published at 14
    req = make_request("p", prompt, H, sha256)
    _complete(manager, req, 15)

    rec = mamba._partial_hit_reqs.get(req.request_id)
    assert rec is not None
    block_idx, source_block = rec
    assert source_block.block_hash is not None
    # the end-of-step pass also left the tail marker naming this block
    # (exactly the shape the 40 prod events printed)
    mamba._producer_partial_tail_reqs[req.request_id] = (source_block, 15)

    # Replay the exact relocation: block moves to the end, slot becomes null.
    blocks = mamba.req_to_blocks[req.request_id]
    blocks.append(blocks[block_idx])
    blocks[block_idx] = mamba._null_block

    manager.allocate_slots(req, 1, 15)  # next (decode) step
    assert mamba._partial_hit_reqs.get(req.request_id) is None
    # the landing state stays published for future replays
    assert source_block.block_hash is not None
