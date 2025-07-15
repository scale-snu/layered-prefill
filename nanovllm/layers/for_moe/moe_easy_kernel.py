import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# These were C++/CUDA kernels in original vllm.

# vllm : ops.topk_softmax
def topk_softmax(
    topk_weights: torch.Tensor,           # [num_tokens, topk] (output, in-place)
    topk_indices: torch.Tensor,           # [num_tokens, topk] (output, in-place)
    token_expert_indices: torch.Tensor,   # [num_tokens, topk] (output, in-place)
    gating_output: torch.Tensor           # [num_tokens, num_experts] (input)
) -> None:
    vals, indices = torch.topk(gating_output, k=topk_weights.size(1), dim=-1)

    weights = torch.softmax(vals, dim=-1)

    topk_weights.copy_(weights)
    topk_indices.copy_(indices)
    token_expert_indices.copy_(indices)

# vllm : ops.moe_sum
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    """
    input: [num_tokens, topk, hidden_size]
    output: [num_tokens, hidden_size]
    """
    torch.sum(input, dim=1, out=output)

# vllm : torch.ops._C.silu_and_mul
def silu_and_mul(out: torch.Tensor, input: torch.Tensor):
    """
    input: [num_tokens, 2*d]
    out: [num_tokens, d]
    """
    assert input.ndim == 2 and out.ndim == 2
    num_tokens, two_d = input.shape
    d = two_d // 2
    assert out.shape == (num_tokens, d)
    x = input[:, :d]
    y = input[:, d:]
    # silu(x) = x * sigmoid(x)
    torch.mul(x * torch.sigmoid(x), y, out=out) 

# silu and mul is appropriate to use triton kernel than topk_softmax and moe_sum
# however I use pytorch-based kernel now for simplicity 
@triton.jit
def silu_and_mul_kernel(
    input_ptr,  # [num_tokens, 2*d]
    output_ptr, # [num_tokens, d]
    num_tokens, # int
    d,          # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row = pid
    if row < num_tokens:
        # x: input[row, :d], y: input[row, d:]
        x_ptr = input_ptr + row * 2 * d + offs
        y_ptr = input_ptr + row * 2 * d + d + offs
        out_ptr = output_ptr + row * d + offs
        mask = offs < d
        x = tl.load(x_ptr, mask=mask, other=0.0)
        y = tl.load(y_ptr, mask=mask, other=0.0)
        silu_x = x / (1.0 + tl.exp(-x))
        out = silu_x * y
        tl.store(out_ptr, out, mask=mask)

def silu_and_mul_triton(output: torch.Tensor, input: torch.Tensor):
    assert input.ndim == 2 and output.ndim == 2
    num_tokens, two_d = input.shape
    d = two_d // 2
    # d보다 크거나 같은 2의 제곱수, 최대 128
    block_size = 1 << (d - 1).bit_length()
    block_size = min(block_size, 128)
    grid = (num_tokens,)
    silu_and_mul_kernel[grid](
        input,
        output,
        num_tokens,
        d,
        BLOCK_SIZE=block_size,
    )

# vllm : torch.ops._C.gelu_and_mul
def gelu_and_mul(out: torch.Tensor, x: torch.Tensor):
    """
    out: (num_tokens, d)
    x: (num_tokens, 2 * d)
    """
    d = x.shape[-1] // 2
    # 앞 절반에 GELU 적용
    x0 = x[..., :d]
    x1 = x[..., d:]
    # PyTorch의 GELU (기본은 'none' approximation)
    gelu_x0 = F.gelu(x0)
    # 곱셈 후 out에 저장 (in-place)
    out.copy_(gelu_x0 * x1)

# gelu and mul is appropriate to use triton kernel than topk_softmax and moe_sum
# however I use pytorch-based kernel now for simplicity 
@triton.jit
def gelu_and_mul_kernel(
    x_ptr,         # input: [num_tokens, 2 * d]
    out_ptr,       # output: [num_tokens, d]
    d: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # x: [num_tokens, 2 * d]
    x0_ptrs = x_ptr + pid * stride_xm + offs * stride_xn
    x1_ptrs = x_ptr + pid * stride_xm + (offs + d) * stride_xn

    mask = offs < d
    x0 = tl.load(x0_ptrs, mask=mask, other=0.0)
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0)

    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    gelu_x0 = 0.5 * x0 * (1.0 + tl.erf(x0 / sqrt_2))

    out = gelu_x0 * x1

    out_ptrs = out_ptr + pid * stride_outm + offs * stride_outn
    tl.store(out_ptrs, out, mask=mask)

def gelu_and_mul_triton(out: torch.Tensor, x: torch.Tensor):
    assert x.ndim == 2
    num_tokens, two_d = x.shape
    d = two_d // 2
    assert out.shape == (num_tokens, d)
    BLOCK_SIZE = triton.next_power_of_2(d)
    grid = (num_tokens,)

    gelu_and_mul_kernel[grid](
        x, out, d,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE
    )