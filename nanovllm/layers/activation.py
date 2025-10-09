import torch
from torch import nn
import torch.nn.functional as F

from nanovllm import ops


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Input tensor must be 2D"
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out
