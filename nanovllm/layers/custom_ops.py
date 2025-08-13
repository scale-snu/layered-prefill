import torch
from nanovllm import ops
from pathlib import Path

from typing import Tuple


@torch.library.custom_op("ops::rotary_embedding", mutates_args=("query", "key"))
def rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int, cos_sin_cache: torch.Tensor, is_neox: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
    return query.clone(), key.clone()

@rotary_embedding.register_fake
def _(positions, query, key, head_size, cos_sin_cache, is_neox):
    return torch.empty_like(query), torch.empty_like(key)

@torch.library.custom_op("ops::rms_norm", mutates_args=("out",))
def rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ops.rms_norm(out, x, weight, eps)
    return out.clone()

@rms_norm.register_fake
def _(out, x, weight, eps):
    return torch.empty_like(out)

@torch.library.custom_op("ops::add_rms_norm", mutates_args=("out",))
def add_rms_norm(out: torch.Tensor, residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ops.add_rms_norm(out, residual, x, weight, eps)
    return out.clone()

@add_rms_norm.register_fake
def _(out, residual, x, weight, eps):
    return torch.empty_like(out)

@torch.compiler.disable(recursive=False)
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    ops.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

