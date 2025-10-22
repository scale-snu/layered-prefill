#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"

namespace sarathi {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  int64_t out_stride_0,
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  int64_t input_stride_0,
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * input_stride_0 + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * input_stride_0 + idx];
    out[blockIdx.x * out_stride_0 + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}


// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel_3d(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  int64_t out_stride_0,
  int64_t out_stride_1,
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  int64_t input_stride_0,
  int64_t input_stride_1,
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int num_heads,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  int64_t token_idx = blockIdx.x / num_heads;
  int64_t head_idx = blockIdx.x % num_heads;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[token_idx * input_stride_0 + head_idx * input_stride_1 + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[token_idx * input_stride_0 + head_idx * input_stride_1 + idx];
    out[token_idx * out_stride_0 + head_idx * out_stride_1 + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void add_rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  int64_t out_stride_0,
  scalar_t* __restrict__ residual,             // [num_tokens, hidden_size]
  int64_t residual_stride_0,
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  int64_t input_stride_0,
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * input_stride_0 + idx] + (float) residual[blockIdx.x * residual_stride_0 + idx];
    residual[blockIdx.x * residual_stride_0 + idx] = (scalar_t) x;
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) residual[blockIdx.x * residual_stride_0 + idx];
    out[blockIdx.x * out_stride_0 + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void add_rms_norm_kernel_3d(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  int64_t out_stride_0,
  int64_t out_stride_1,
  scalar_t* __restrict__ residual,             // [num_tokens, hidden_size]
  int64_t residual_stride_0,
  int64_t residual_stride_1,
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  int64_t input_stride_0,
  int64_t input_stride_1,
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int num_heads,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  int64_t token_idx = blockIdx.x / num_heads;
  int64_t head_idx = blockIdx.x % num_heads;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[token_idx * input_stride_0 + head_idx * input_stride_1 + idx] + (float) residual[token_idx * residual_stride_0 + head_idx * residual_stride_1 + idx];
    residual[token_idx * residual_stride_0 + head_idx * residual_stride_1 + idx] = (scalar_t) x;
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) residual[token_idx * residual_stride_0 + head_idx * residual_stride_1 + idx];
    out[token_idx * out_stride_0 + head_idx * out_stride_1 + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace sarathi

void rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size] or [num_tokens, num_head, hidden_size]
  torch::Tensor& input,    // [num_tokens, hidden_size] or [num_tokens, num_head, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  if (out.dim() == 2) {
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    SARATHI_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "rms_norm_kernel",
      [&] {
        sarathi::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(),
          out.stride(0),
          input.data_ptr<scalar_t>(),
          input.stride(0),
          weight.data_ptr<scalar_t>(),
          epsilon,
          num_tokens,
          hidden_size);
      });
  } else {
    int num_tokens = input.size(0);
    int num_heads = input.size(1);
    int hidden_size = input.size(2);

    dim3 grid(num_tokens * num_heads);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    SARATHI_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "rms_norm_kernel_3d",
      [&] {
        sarathi::rms_norm_kernel_3d<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(),
          out.stride(0),
          out.stride(1),
          input.data_ptr<scalar_t>(),
          input.stride(0),
          input.stride(1),
          weight.data_ptr<scalar_t>(),
          epsilon,
          num_tokens,
          num_heads,
          hidden_size);
      });
  }
}

void add_rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size] or [num_tokens, num_head, hidden_size]
  torch::Tensor& residual, // [num_tokens, hidden_size] or [num_tokens, num_head, hidden_size]
  torch::Tensor& input,    // [num_tokens, hidden_size] or [num_tokens, num_head, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  if (out.dim() == 2) {
    int num_tokens = input.size(0);
    int hidden_size = input.size(1);

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    SARATHI_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "add_rms_norm_kernel",
      [&] {
        sarathi::add_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(),
          out.stride(0),
          residual.data_ptr<scalar_t>(),
          residual.stride(0),
          input.data_ptr<scalar_t>(),
          input.stride(0),
          weight.data_ptr<scalar_t>(),
          epsilon,
          num_tokens,
          hidden_size);
      });
  } else {
    int num_tokens = input.size(0);
    int num_heads = input.size(1);
    int hidden_size = input.size(2);

    dim3 grid(num_tokens * num_heads);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    SARATHI_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "add_rms_norm_kernel_3d",
      [&] {
        sarathi::add_rms_norm_kernel_3d<scalar_t><<<grid, block, 0, stream>>>(
          out.data_ptr<scalar_t>(),
          out.stride(0),
          out.stride(1),
          residual.data_ptr<scalar_t>(),
          residual.stride(0),
          residual.stride(1),
          input.data_ptr<scalar_t>(),
          input.stride(0),
          input.stride(1),
          weight.data_ptr<scalar_t>(),
          epsilon,
          num_tokens,
          num_heads,
          hidden_size);
      });
  }
}
