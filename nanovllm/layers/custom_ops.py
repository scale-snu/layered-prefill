import torch
from nanovllm import ops
from nanovllm.utils.scalar_type import ScalarType
from pathlib import Path

from typing import Tuple


# @torch.library.custom_op("ops::rotary_embedding", mutates_args=("query", "key"))
@torch.compiler.disable(recursive=False)
def rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int, cos_sin_cache: torch.Tensor, is_neox: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
    return query, key

# @rotary_embedding.register_fake
# def _(positions, query, key, head_size, cos_sin_cache, is_neox):
#     return torch.empty_like(query), torch.empty_like(key)

# @torch.library.custom_op("ops::silu_and_mul", mutates_args=("out",))
@torch.compiler.disable(recursive=False)
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ops.silu_and_mul(out, x)
    return out

# @silu_and_mul.register_fake
# def _(out, x):
#     return torch.empty_like(out)

@torch.compiler.disable(recursive=False)
def swigluoai_and_mul(out: torch.Tensor, x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    ops.swigluoai_and_mul(out, x, alpha, limit)
    return out

# @torch.library.custom_op("ops::rms_norm", mutates_args=("out",))
@torch.compiler.disable(recursive=False)
def rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ops.rms_norm(out, x, weight, eps)
    return out

# @rms_norm.register_fake
# def _(out, x, weight, eps):
#     return torch.empty_like(out)

# @torch.library.custom_op("ops::add_rms_norm", mutates_args=("out",))
@torch.compiler.disable(recursive=False)
def add_rms_norm(out: torch.Tensor, residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ops.add_rms_norm(out, residual, x, weight, eps)
    return out

# @add_rms_norm.register_fake
# def _(out, residual, x, weight, eps):
#     return torch.empty_like(out)

@torch.compiler.disable(recursive=False)
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    ops.store_kvcache(key, value, k_cache, v_cache, slot_mapping)



def moe_wna16_marlin_gemm(input: torch.Tensor, output: torch.Tensor | None,
                          b_qweight: torch.Tensor,
                          b_bias: torch.Tensor | None,
                          b_scales: torch.Tensor,
                          global_scale: torch.Tensor | None,
                          b_qzeros: torch.Tensor | None,
                          g_idx: torch.Tensor | None,
                          perm: torch.Tensor | None,
                          workspace: torch.Tensor,
                          sorted_token_ids: torch.Tensor,
                          expert_ids: torch.Tensor,
                          num_tokens_past_padded: torch.Tensor,
                          topk_weights: torch.Tensor, moe_block_size: int,
                          top_k: int, mul_topk_weights: bool, is_ep: bool,
                          b_q_type: ScalarType, size_m: int, size_n: int,
                          size_k: int, is_k_full: bool, use_atomic_add: bool,
                          use_fp32_reduce: bool,
                          is_zp_float: bool) -> torch.Tensor:
    return ops.moe_wna16_marlin_gemm(
        input, output, b_qweight, b_bias, b_scales, global_scale, b_qzeros,
        g_idx, perm, workspace, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, is_ep, b_q_type.id, size_m, size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float)
