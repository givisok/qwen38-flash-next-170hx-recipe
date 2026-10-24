# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp precision switches, ported from Morrowmake/vllm-cmp170hx (ampere).

``VLLM_QWEN4_ROUTER_FP32`` makes the MoE router GEMM return FP32 logits via
GateLinear's cuBLAS bf16x bf16 -> fp32 epilogue tier (works on any CUDA
device; the bf16 rounding of router logits otherwise merges nearby expert
scores and hands ties to the top-k). ``VLLM_QWEN4_MTP_PER_STREAM_NORM``
normalizes the MTP hidden input per hyper-connection stream instead of over
the flattened ``[hc_count * hidden]`` stack. Both are read from the
environment at model construction — never in the hot path.
"""

import os

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

logger = init_logger(__name__)


def _router_fp32() -> bool:
    return os.environ.get("VLLM_QWEN4_ROUTER_FP32", "0") == "1"


def _mtp_per_stream_norm() -> bool:
    return os.environ.get("VLLM_QWEN4_MTP_PER_STREAM_NORM", "0") == "1"


def apply_router_precision(gate: nn.Module) -> None:
    """Select the router logits dtype for a Qwen4Exp sparse MoE block."""
    if _router_fp32():
        gate.set_out_dtype(torch.float32)
    logger.info_once(
        "Qwen4Exp router logits: %s (VLLM_QWEN4_ROUTER_FP32=%d)",
        "FP32" if _router_fp32() else "model dtype",
        int(_router_fp32()),
    )


class PerStreamGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm over ``[..., hc_count * hidden]`` that normalizes every
    hyper-connection stream on its own, with that stream's slice of the
    checkpoint's affine weight. The parameter is the flat norm's, so weights
    load unchanged.
    """

    def __init__(self, hidden_size: int, hc_count: int, eps: float) -> None:
        super().__init__(hidden_size * hc_count, eps=eps)
        self.hc_count = hc_count
        self.stream_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Like GemmaRMSNorm, apply (1 + weight) in FP32 before one cast back.
        streams = x.unflatten(-1, (self.hc_count, self.stream_size)).float()
        variance = streams.square().mean(dim=-1, keepdim=True)
        streams = streams * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.float().view(self.hc_count, self.stream_size)
        return (streams * (1.0 + weight)).to(x.dtype).flatten(-2)


def mtp_hidden_norm(hidden_size: int, hc_count: int, eps: float) -> GemmaRMSNorm:
    """Pre-FC norm of the MTP hidden input ``[T, hc_count * hidden]``: the flat
    GemmaRMSNorm, or the per-stream one under
    ``VLLM_QWEN4_MTP_PER_STREAM_NORM``.
    """
    per_stream = _mtp_per_stream_norm()
    logger.info_once(
        "Qwen4Exp MTP hidden norm: %s (VLLM_QWEN4_MTP_PER_STREAM_NORM=%d)",
        "per-stream" if per_stream else "flat",
        int(per_stream),
    )
    if per_stream:
        return PerStreamGemmaRMSNorm(hidden_size, hc_count, eps=eps)
    return GemmaRMSNorm(hidden_size * hc_count, eps=eps)
