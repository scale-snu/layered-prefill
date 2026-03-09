/*
 * Adapted from https://github.com/vllm-project/vllm
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

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
