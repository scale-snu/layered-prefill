import torch
from torch import nn

from nanovllm.layers import custom_ops as ops


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        out = torch.empty_like(x)
        if residual is None:
            ops.rms_norm(out, x, self.weight.data, self.eps)
            return out.reshape(shape)
        else:
            ops.add_rms_norm(out, residual, x, self.weight.data, self.eps)
            return out.reshape(shape), residual
