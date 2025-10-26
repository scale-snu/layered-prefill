import torch
from torch import nn
import triton
import triton.language as tl

from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.layers import custom_ops as ops


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    ops.store_kvcache(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        window_size: int = -1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size

        self.k_cache = self.v_cache = torch.tensor([])

        self.fa_version = 3 if torch.cuda.get_device_capability()[0] >= 9 else 2

        self.cu_seqlens_q_cache = torch.arange(0, 8192 + 1, device='cuda', dtype=torch.int32)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: None | torch.Tensor = None) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()

        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(
                k,
                v,
                k_cache,
                v_cache,
                context.slot_mapping,
            )

        o = torch.empty((q.size(0), self.num_heads, self.head_dim), device=q.device, dtype=q.dtype)

        return self.forward_attention(o, q, k, v, sinks)

    @torch.compile(dynamic=True)
    def forward_attention(self, o: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: None | torch.Tensor = None) -> torch.Tensor:
        context = get_context()

        k_cache, v_cache = self.k_cache, self.v_cache

        if context.is_prefill and context.len_prefill > 0:
            if context.prefill_block_tables is not None:
                k, v = k_cache, v_cache
            else:
                k = k[:context.len_prefill]
                v = v[:context.len_prefill]
            flash_attn_varlen_func(
                q[:context.len_prefill], k, v,
                out=o[:context.len_prefill],
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k if context.prefill_block_tables is None else None,
                seqused_k=context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1] if context.prefill_block_tables is not None else None,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.prefill_block_tables,
                window_size=(self.window_size, -1),
                fa_version=self.fa_version,
                s_aux=sinks,
                num_splits=32 if self.fa_version == 3 else 1,
            )

        if context.decode_block_tables is not None:
            flash_attn_varlen_func(
                q[context.len_prefill:], k_cache, v_cache,
                out=o[context.len_prefill:],
                max_seqlen_q=1,
                cu_seqlens_q=self.cu_seqlens_q_cache[:context.decode_block_tables.size(0) + 1],
                max_seqlen_k=context.max_seqlen_k_dec,
                seqused_k=context.context_lens,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.decode_block_tables,
                window_size=(self.window_size, -1),
                fa_version=self.fa_version,
                s_aux=sinks,
                num_splits=32 if self.fa_version == 3 else 1,
            )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
