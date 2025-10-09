#include <torch/extension.h>

#include <vector>
#include <tuple>
#include <cstdint>

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