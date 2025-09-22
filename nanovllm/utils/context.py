from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    len_prefill: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    prefill_block_tables: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    prefill_compute_layers: list[int] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
        is_prefill=False,
        len_prefill=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        slot_mapping=None,
        context_lens=None,
        prefill_block_tables=None,
        decode_block_tables=None,
        prefill_compute_layers=None,
        ):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        len_prefill=len_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        prefill_block_tables=prefill_block_tables,
        decode_block_tables=decode_block_tables,
        prefill_compute_layers=prefill_compute_layers,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
