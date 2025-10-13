from typing import List, Optional

import torch
import torch.distributed as dist

from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from nanovllm.config import Config
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.engine.sequence import Sequence, SequenceStatus


class FlashinferAttentionWrapper:
    def __init__(
        self,
        config: Config,
        device: torch.device,
    ):
        tp_size = dist.get_world_size()
        self.num_layers = config.hf_config.num_hidden_layers
        self.device = device
        self.num_q_heads = config.hf_config.num_attention_heads // tp_size
        self.num_kv_heads = config.hf_config.num_key_value_heads // tp_size
        self.head_dim = getattr(config.hf_config, 'head_dim', None) or config.hf_config.hidden_size // config.hf_config.num_attention_heads
        self.dtype = config.hf_config.dtype
        self.block_size = config.kvcache_block_size
        self.num_gpu_blocks: int = -1

        prefill_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            prefill_workspace_buffer, "NHD"
        )

        decode_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            decode_workspace_buffer, "NHD"
        )

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        self.append_qo_indptr_tensor = None
        self.append_kv_page_indices_tensor = None
        self.append_kv_page_indptr_tensor = None
        self.append_kv_last_page_len_tensor = None
        self.append_batch_indices_tensor = None  # newly increased
        self.append_positions_tensor = None      # newly increased

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    def init_gpu_cache(self, num_gpu_blocks: int) -> None:
        gpu_cache: List[torch.Tensor] = []
        self.num_gpu_blocks = num_gpu_blocks

        for _ in range(self.num_layers):
            gpu_blocks = self.get_cache_block(
                self.num_gpu_blocks, dtype=self.dtype, device="cuda"
            )
            gpu_cache.append(gpu_blocks)

        self.gpu_cache = gpu_cache

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def begin_forward(
        self,
        seqs: list[Sequence],
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        prefill_qo_indptr: List[int] = [0]
        decode_qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        prefill_kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        prefill_kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        # Since the prefill_kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the prefill_kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        prefill_kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]

        # newly increase batch_indices and positions
        prefill_batch_indices: List[int] = []
        decode_batch_indices: List[int] = []
        prefill_positions: List[int] = []
        decode_positions: List[int] = []

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        for seq_idx, seq in enumerate(seqs):
            if not seq.status == SequenceStatus.PREFILLING:
                continue

            # ONLY used for profiling
            if not seq.block_table:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            self.contains_prefill = True

            prompt_chunk_len = seq.num_tokens_to_process
            processed_prompt_len = seq.num_processed_tokens

            current_total_len = processed_prompt_len + prompt_chunk_len

            # indptr for the prompt tokens in q/o tensor
            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            prefill_kv_page_indices.extend(seq.block_table[:num_blocks_in_use])
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )
            # computing batch_indices and positions
            prefill_batch_indices.extend([seq_idx] * prompt_chunk_len)
            prefill_positions.extend(range(processed_prompt_len, current_total_len))

        for seq_idx, seq in enumerate(seqs):
            if not seq.status == SequenceStatus.DECODING:
                continue

            if not seq.block_table:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = len(seq)
            # indptr for the prompt tokens in q/o tensor
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size
            decode_kv_page_indices.extend(seq.block_table[:num_blocks_in_use])
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )
            # computing batch_indices and positions
            decode_batch_indices.append(seq_idx)
            decode_positions.append(context_len - 1)  # Only the latest token is appended when decoding

        if self.contains_prefill:
            self.prefill_wrapper.begin_forward(
                self.to_int_tensor(prefill_qo_indptr),
                self.to_int_tensor(prefill_kv_page_indptr),
                self.to_int_tensor(prefill_kv_page_indices),
                self.to_int_tensor(prefill_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

        if self.contains_decode:
            self.decode_wrapper.begin_forward(
                self.to_int_tensor(decode_qo_indptr),
                self.to_int_tensor(decode_kv_page_indptr),
                self.to_int_tensor(decode_kv_page_indices),
                self.to_int_tensor(decode_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1

        self.append_qo_indptr_tensor = self.to_int_tensor(
            prefill_qo_indptr[:-1]
            + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]
        )
        self.append_kv_page_indices_tensor = self.to_int_tensor(
            prefill_kv_page_indices + decode_kv_page_indices
        )
        self.append_kv_page_indptr_tensor = self.to_int_tensor(
            prefill_kv_page_indptr[:-1]
            + [x + prefill_kv_page_indptr[-1] for x in decode_kv_page_indptr]
        )
        self.append_kv_last_page_len_tensor = self.to_int_tensor(
            prefill_kv_last_page_len + decode_kv_last_page_len
        )
        self.append_batch_indices_tensor = self.to_int_tensor(
            prefill_batch_indices + decode_batch_indices
        )
        self.append_positions_tensor = self.to_int_tensor(
            prefill_positions + decode_positions
        )

    def end_forward(self):
        if self.contains_prefill:
            self.prefill_wrapper.end_forward()

        if self.contains_decode:
            self.decode_wrapper.end_forward()

        self.is_metadata_initialized = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
        key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
        value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)

        append_paged_kv_cache(
            append_key=key,
            append_value=value,
            batch_indices=self.append_batch_indices_tensor,
            positions=self.append_positions_tensor,
            paged_kv_cache=self.gpu_cache[layer_cache_idx],
            kv_indices=self.append_kv_page_indices_tensor,
            kv_indptr=self.append_kv_page_indptr_tensor,
            kv_last_page_len=self.append_kv_last_page_len_tensor,
            kv_layout="NHD",
        )

        if self.contains_prefill:
            self.prefill_wrapper._sm_scale = softmax_scale
            output[: self.num_prefill_tokens] = self.prefill_wrapper.run(
                query[: self.num_prefill_tokens],
                self.gpu_cache[layer_cache_idx],
                # pos_encoding_mode="NONE",
                # sm_scale=softmax_scale,
                # causal=True,
            )

        if self.contains_decode:
            self.decode_wrapper._sm_scale = softmax_scale
            output[self.num_prefill_tokens : self.num_total_tokens] = (
                self.decode_wrapper.run(
                    query[self.num_prefill_tokens : self.num_total_tokens],
                    self.gpu_cache[layer_cache_idx],
                    # pos_encoding_mode="NONE",
                    # sm_scale=softmax_scale,
                    # causal=True,
                )
            )

        output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
