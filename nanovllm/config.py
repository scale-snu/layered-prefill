import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig

HF_HOME = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(HF_HOME, "hub"))

def hf_snapshot_dir(
    repo_id: str,
    *,
    cache_dir: str | os.PathLike = HF_CACHE_DIR,
    revision: Optional[str] = None,
    require_exists: bool = True,
) -> str:
    """
    repo_id 예: "openai/gpt-oss-20b"
    반환 예:
      /data/cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/<commit_hash>/

    revision을 주면 그 해시(또는 refs 경로로 해석되는 값)를 우선 사용.
    require_exists=True면 실제 존재하는 경로만 반환하고 없으면 FileNotFoundError.
    """
    cache_dir = Path(cache_dir)

    if "/" not in repo_id:
        raise ValueError(f"repo_id는 'org/name' 형태여야 합니다. 입력: {repo_id!r}")

    org, name = repo_id.split("/", 1)
    model_root = cache_dir / f"models--{org}--{name}"
    snapshots_dir = model_root / "snapshots"

    # 1) revision을 지정한 경우
    if revision:
        candidate = snapshots_dir / revision
        if require_exists and not candidate.is_dir():
            raise FileNotFoundError(f"지정한 revision 경로가 없습니다: {candidate}")
        return str(candidate)

    # 2) revision 미지정이면 refs/main -> commit 해시를 따라가보기
    refs_main = model_root / "refs" / "main"
    if refs_main.is_file():
        commit = refs_main.read_text(encoding="utf-8").strip()
        if commit:
            candidate = snapshots_dir / commit
            if (not require_exists) or candidate.is_dir():
                return str(candidate)

    # 3) 마지막으로 snapshots 아래 디렉터리 중 가장 최신 mtime 선택
    if snapshots_dir.is_dir():
        dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if dirs:
            latest = max(dirs, key=lambda p: p.stat().st_mtime)
            return str(latest)

    raise FileNotFoundError(
        f"스냅샷 폴더를 찾지 못했습니다. repo_id={repo_id!r}, cache_dir={str(cache_dir)!r}"
    )


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
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    # Scheduling mode settings
    # "chunked-prefill": Split prompt into chunks (default)
    # "orca": Process the entire prompt at once
    # "layered-prefill": Split prompt into multiple stages
    schedule_mode: str = "chunked-prefill" # or "orca" or "layered-prefill"
    # Number of stages to use in layered-prefill mode
    # Each stage creates a separate queue to manage sequences by stage
    # Example: num_stages=4 means processing in 4 stages
    num_stages: int = 1
    rpc_base_path: str = "/tmp"
    moe_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if not os.path.isdir(self.model):
            # Try to get from HuggingFace Hub
            self.model = hf_snapshot_dir(self.model)
            print(f"Model path from HF Hub: {self.model}")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.torch_dtype = self.hf_config.torch_dtype or torch.bfloat16
        self.hf_config.moe_dtype = self.moe_dtype
        self.hf_config.max_num_batched_tokens = self.max_num_batched_tokens
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len
