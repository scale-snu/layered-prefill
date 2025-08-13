#include <torch/extension.h>

#include <vector>
#include <tuple>
#include <cstdint>


void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void add_rms_norm(
  torch::Tensor& out,
  torch::Tensor& residual,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void swigluoai_and_mul(
  torch::Tensor& out,
  torch::Tensor& input,
  const float alpha, const float limit);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

void topk_softmax(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output
);

void moe_align_block_size(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad
);

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

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
fptr_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(fptr_t buffer);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rotary_embedding",
    &rotary_embedding,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");
  m.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  m.def(
    "add_rms_norm",
    &add_rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  m.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  m.def(
    "swigluoai_and_mul",
    &swigluoai_and_mul,
    "Activation function used in SwiGLU (OpenAI variant).");
  m.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  m.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");
  m.def(
    "topk_softmax",
    &topk_softmax,
    "Apply topk softmax to the gating outputs.");
  m.def(
    "moe_align_block_size",
    &moe_align_block_size,
    "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");
  m.def(
    "store_kvcache",
    &store_kvcache,
    "Store KV cache (CUDA)");
  // Custom all-reduce kernels
  m.def("init_custom_ar",
        &init_custom_ar,
        "Initialize custom all-reduce with fake IPC pointers and rank data.");
  m.def(
      "all_reduce",
      &all_reduce,
      "Perform all-reduce operation on the input tensor using the custom all-reduce handle.");

  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);

  m.def("register_buffer", &register_buffer);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);

  m.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  m.def("open_mem_handle", &open_mem_handle);

  m.def("free_shared_buffer", &free_shared_buffer);
}
