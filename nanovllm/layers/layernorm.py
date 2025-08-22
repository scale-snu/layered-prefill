import torch
from torch import nn

from nanovllm import layernorm_ops


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
            layernorm_ops.rms_norm(out, x, self.weight.data, self.eps)
            return out.reshape(shape)
        else:
            x += residual
            residual = x.clone()
            layernorm_ops.rms_norm(out, x, self.weight.data, self.eps)
            return out.reshape(shape), residual
