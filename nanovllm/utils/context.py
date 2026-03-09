# MIT License
# Adapted from https://github.com/GeeeekExplorer/nanovllm
# Copyright (c) 2025 GeeeekExplorer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    max_seqlen_k_dec: int = 0
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
        max_seqlen_k_dec=0,
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
        max_seqlen_k_dec=max_seqlen_k_dec,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        prefill_block_tables=prefill_block_tables,
        decode_block_tables=decode_block_tables,
        prefill_compute_layers=prefill_compute_layers,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
