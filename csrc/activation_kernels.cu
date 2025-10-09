#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "dispatch_utils.h"

namespace sarathi {

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [num_tokens, d]
  const scalar_t* __restrict__ input,       // [num_tokens, 2, d]
  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
  scalar_t* __restrict__ out,          // [..., d]
  const scalar_t* __restrict__ input,  // [..., 2, d]
  const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  // TODO: Vectorize loads and stores.
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // gate = x[..., ::2]  (even indices)
    const scalar_t gate = __ldg(&input[token_idx * 2 * d + 2 * idx]);
    // up = x[..., 1::2]   (odd indices)
    const scalar_t up = __ldg(&input[token_idx * 2 * d + 2 * idx + 1]);

    out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
  }
}

} // namespace sarathi

void silu_and_mul(
  torch::Tensor& out,      // [num_tokens, d]
  torch::Tensor& input)    // [num_tokens, 2 * d]
{
  int64_t num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  SARATHI_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "silu_and_mul_kernel",
    [&] {
      sarathi::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d);
    });
}

void swigluoai_and_mul(
  torch::Tensor& out,      // [num_tokens, d]
  torch::Tensor& input,
  const float alpha=1.702, const float limit=7.0)    // [num_tokens, 2 * d]
{
  int64_t num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  SARATHI_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "swigluoai_and_mul_kernel",
    [&] {
      sarathi::swigluoai_and_mul_kernel<scalar_t, sarathi::swigluoai_and_mul><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d, alpha, limit);
    });
}

namespace sarathi {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
  scalar_t* __restrict__ out,               // [num_tokens, d]
  const scalar_t* __restrict__ input,       // [num_tokens, d]
  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

} // namespace sarathi

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                                  \
  int64_t num_tokens = input.size(0);                                                         \
  int d = input.size(1);                                                                  \
  dim3 grid(num_tokens);                                                                  \
  dim3 block(std::min(d, 1024));                                                          \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                           \
  SARATHI_DISPATCH_FLOATING_TYPES(                                                           \
    input.scalar_type(),                                                                  \
    "activation_kernel",                                                                  \
    [&] {                                                                                 \
      sarathi::activation_kernel<scalar_t, KERNEL<scalar_t>><<<grid, block, 0, stream>>>(    \
        out.data_ptr<scalar_t>(),                                                         \
        input.data_ptr<scalar_t>(),                                                       \
        d);                                                                               \
    });

namespace sarathi {

template<typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float) (x * x * x);
  const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template<typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float) x;
  const T t = (T) tanhf(((T) (f * 0.79788456f)) * (((T) 1.0) + (T) (0.044715f * f) * x));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

} // namespace sarathi

void gelu_new(
  torch::Tensor& out,     // [num_tokens, d]
  torch::Tensor& input)   // [num_tokens, d]
{
  LAUNCH_ACTIVATION_KERNEL(sarathi::gelu_new_kernel);
}

void gelu_fast(
  torch::Tensor& out,     // [num_tokens, d]
  torch::Tensor& input)   // [num_tokens, d]
{
  LAUNCH_ACTIVATION_KERNEL(sarathi::gelu_fast_kernel);
}