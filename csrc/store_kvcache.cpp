#include <torch/extension.h>


void store_kvcache_cuda(
    at::Tensor key, at::Tensor value,
    at::Tensor k_cache, at::Tensor v_cache,
    at::Tensor slot_mapping);


void store_kvcache(
    at::Tensor key, at::Tensor value,
    at::Tensor k_cache, at::Tensor v_cache,
    at::Tensor slot_mapping)
{
    TORCH_CHECK(key.is_cuda(), "CUDA tensors expected.");
    store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("store_kvcache", &store_kvcache, "Store KV cache (CUDA)");
}
