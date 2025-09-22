#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>      // getCurrentCUDAStream
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>         // c10::cuda::CUDAGuard

namespace sarathi {

template <typename scalar_t, typename index_t>
__global__ void store_kvcache_kernel(
    const scalar_t* __restrict__ key,     // [N, Hk, Hd]
    int64_t k_s0, int64_t k_s1, int64_t k_s2,
    const scalar_t* __restrict__ value,   // [N, Hk, Hd]
    int64_t v_s0, int64_t v_s1, int64_t v_s2,
    scalar_t* __restrict__ k_cache,       // [B, BS, Hk, Hd]
    int64_t kc_s0, int64_t kc_s1, int64_t kc_s2, int64_t kc_s3,
    scalar_t* __restrict__ v_cache,       // [B, BS, Hk, Hd]
    int64_t vc_s0, int64_t vc_s1, int64_t vc_s2, int64_t vc_s3,
    const index_t* __restrict__ slot_mapping, // [N], linear slot in [0, B*BS)
    int64_t N,
    int64_t Hk,
    int64_t Hd,
    int64_t D,        // Hk*Hd
    int64_t BS        // block_size
){
    // grid over N (tokens)
    for (int64_t n = blockIdx.x; n < N; n += gridDim.x) {
        const index_t slot = slot_mapping[n];
        const int64_t b  = static_cast<int64_t>(slot) / BS;   // block idx
        const int64_t off = static_cast<int64_t>(slot) - b * BS; // offset within block

        // threads cover D (= Hk*Hd)
        for (int64_t t = threadIdx.x; t < D; t += blockDim.x) {
            const int64_t h  = t / Hd;   // head index
            const int64_t hd = t - h * Hd;

            const scalar_t k_val = key  [n * k_s0 + h * k_s1 + hd * k_s2];
            const scalar_t v_val = value[n * v_s0 + h * v_s1 + hd * v_s2];

            // k_cache[b, off, h, hd]
            k_cache[b * kc_s0 + off * kc_s1 + h * kc_s2 + hd * kc_s3] = k_val;
            v_cache[b * vc_s0 + off * vc_s1 + h * vc_s2 + hd * vc_s3] = v_val;
        }
    }
}

template <typename scalar_t, typename index_t>
void launch_kernel(
    const scalar_t* key, int64_t k_s0, int64_t k_s1, int64_t k_s2,
    const scalar_t* value, int64_t v_s0, int64_t v_s1, int64_t v_s2,
    scalar_t* k_cache, int64_t kc_s0, int64_t kc_s1, int64_t kc_s2, int64_t kc_s3,
    scalar_t* v_cache, int64_t vc_s0, int64_t vc_s1, int64_t vc_s2, int64_t vc_s3,
    const index_t* slot_mapping,
    int64_t N, int64_t Hk, int64_t Hd, int64_t D, int64_t BS,
    cudaStream_t stream)
{
    int threads = (int)(D < 256 ? D : 256);
    if (threads <= 0) threads = 1;
    int grid = (int)(N < 65535 ? N : 65535);

    store_kvcache_kernel<scalar_t, index_t><<<grid, threads, 0, stream>>>(
        key, k_s0, k_s1, k_s2,
        value, v_s0, v_s1, v_s2,
        k_cache, kc_s0, kc_s1, kc_s2, kc_s3,
        v_cache, vc_s0, vc_s1, vc_s2, vc_s3,
        slot_mapping,
        N, Hk, Hd, D, BS
    );
}

} // namespace sarathi

void store_kvcache_cuda(
    at::Tensor key,          // [N, Hk, Hd]
    at::Tensor value,        // [N, Hk, Hd]
    at::Tensor k_cache,      // [B, BS, Hk, Hd]
    at::Tensor v_cache,      // [B, BS, Hk, Hd]
    at::Tensor slot_mapping  // [N], linear slot = block*BS + offset
){
    TORCH_CHECK(key.is_cuda() && value.is_cuda() &&
                k_cache.is_cuda() && v_cache.is_cuda() &&
                slot_mapping.is_cuda(), "All tensors must be CUDA.");

    TORCH_CHECK(key.scalar_type() == value.scalar_type() &&
                key.scalar_type() == k_cache.scalar_type() &&
                key.scalar_type() == v_cache.scalar_type(),
                "key/value/k_cache/v_cache must have same dtype.");

    TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key/value must be [N, Hk, Hd].");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be [B, BS, Hk, Hd].");  // num_blocks, block_size, num_heads, head_dim
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be [N].");

    const int64_t N  = key.size(0);
    const int64_t Hk = key.size(1);
    const int64_t Hd = key.size(2);
    TORCH_CHECK(value.sizes() == key.sizes(), "value shape must match key shape.");

    // key/value layout: last dim contiguous, and stride(1) == Hd
    TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
                "key/value last dim must be contiguous.");
    TORCH_CHECK(key.stride(1) == Hd && value.stride(1) == Hd,
                "key/value stride(1) must equal head_dim (Hd).");

    const int64_t B  = k_cache.size(0);
    const int64_t BS = k_cache.size(1);
    TORCH_CHECK(k_cache.size(2) == Hk && k_cache.size(3) == Hd, "k_cache last dims must be [Hk, Hd].");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must match k_cache shape.");

    // cache: last dim contiguous
    TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
                "k_cache/v_cache last dim (Hd) must be contiguous.");

    TORCH_CHECK(slot_mapping.size(0) == N, "slot_mapping size must be N.");
    // 간단한 상한 체크(가능하면 호출 전 보장)
    TORCH_CHECK(B > 0 && BS > 0, "Invalid cache shape.");
    // (선택) 필요한 경우 slot 상한을 런타임에 검사하고 싶다면 아래 주석 해제
    // auto max_slot = B * BS;
    // TORCH_CHECK((slot_mapping.max().item<int64_t>() < max_slot), "slot out of range.");

    c10::cuda::CUDAGuard device_guard(key.get_device());
    auto stream = c10::cuda::getCurrentCUDAStream();

    const int64_t D = Hk * Hd;
    auto st = key.scalar_type();

    if (slot_mapping.scalar_type() == at::kInt) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, st, "store_kvcache_cuda", [&] {
            sarathi::launch_kernel<scalar_t, int32_t>(
                key.data_ptr<scalar_t>(),
                key.stride(0), key.stride(1), key.stride(2),
                value.data_ptr<scalar_t>(),
                value.stride(0), value.stride(1), value.stride(2),
                k_cache.data_ptr<scalar_t>(),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                v_cache.data_ptr<scalar_t>(),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
                slot_mapping.data_ptr<int32_t>(),
                N, Hk, Hd, D, /*BS=*/k_cache.size(1),
                stream.stream()
            );
        });
    } else if (slot_mapping.scalar_type() == at::kLong) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, st, "store_kvcache_cuda", [&] {
            sarathi::launch_kernel<scalar_t, int64_t>(
                key.data_ptr<scalar_t>(),
                key.stride(0), key.stride(1), key.stride(2),
                value.data_ptr<scalar_t>(),
                value.stride(0), value.stride(1), value.stride(2),
                k_cache.data_ptr<scalar_t>(),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
                v_cache.data_ptr<scalar_t>(),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
                slot_mapping.data_ptr<int64_t>(),
                N, Hk, Hd, D, /*BS=*/k_cache.size(1),
                stream.stream()
            );
        });
    } else {
        TORCH_CHECK(false, "slot_mapping must be int32 or int64.");
    }

    C10_CUDA_CHECK(cudaGetLastError());
}
