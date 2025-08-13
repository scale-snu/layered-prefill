import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    nccl_port: int = 2333
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Scheduling mode settings
    # "chunked-prefill": Split prompt into chunks (default)
    # "orca": Process the entire prompt at once
    # "layered-prefill": Split prompt into multiple stages
    schedule_mode: str = "chunked-prefill" # or "orca" or "layered-prefill"
    # Number of stages to use in layered-prefill mode
    # Each stage creates a separate queue to manage sequences by stage
    # Example: num_stages=4 means processing in 4 stages
    num_stages: int = 4
    rpc_base_path: str = "/tmp"

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.torch_dtype = self.hf_config.torch_dtype or torch.bfloat16
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len
