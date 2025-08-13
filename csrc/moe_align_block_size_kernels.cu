#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include "dispatch_utils.h"

#define WARP_SIZE 32
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#define SARATHI_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)

// namespace sarathi {

// namespace {
// __device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
//                                          int32_t col) {
//   // don't worry about overflow because num_experts is relatively small
//   return row * total_col + col;
// }
// }  // namespace

// template <typename scalar_t>
// __global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids,
//                                             int32_t* sorted_token_ids,
//                                             int32_t* expert_ids,
//                                             int32_t* total_tokens_post_pad,
//                                             int32_t num_experts,
//                                             int32_t block_size, size_t numel) {
//   const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
//   const size_t start_idx = threadIdx.x * tokens_per_thread;

//   extern __shared__ int32_t shared_mem[];

//   int32_t* tokens_cnts =
//       shared_mem;  // 2d tensor with shape (num_experts + 1, num_experts)
//   int32_t* cumsum =
//       shared_mem + (num_experts + 1) *
//                        num_experts;  // 1d tensor with shape (num_experts + 1)

//   for (int i = 0; i < num_experts; ++i) {
//     tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
//   }

//   /**
//    * In the first step we compute token_cnts[thread_index + 1][expert_index],
//    * which counts how many tokens in the token shard of thread_index are
//    * assigned to expert expert_index.
//    */
//   for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
//     ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
//   }

//   __syncthreads();

//   // For each expert we accumulate the token counts from the different threads.
//   tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
//   for (int i = 1; i <= blockDim.x; ++i) {
//     tokens_cnts[index(num_experts, i, threadIdx.x)] +=
//         tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
//   }

//   __syncthreads();

//   // We accumulate the token counts of all experts in thread 0.
//   if (threadIdx.x == 0) {
//     cumsum[0] = 0;
//     for (int i = 1; i <= num_experts; ++i) {
//       cumsum[i] = cumsum[i - 1] +
//                   CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
//                           block_size) *
//                       block_size;
//     }
//     *total_tokens_post_pad = cumsum[num_experts];
//   }

//   __syncthreads();

//   /**
//    * For each expert, each thread processes the tokens of the corresponding
//    * blocks and stores the corresponding expert_id for each block.
//    */
//   for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
//        i += block_size) {
//     expert_ids[i / block_size] = threadIdx.x;
//   }

//   /**
//    * Each thread processes a token shard, calculating the index of each token
//    * after sorting by expert number. Given the example topk_ids =
//    * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
//    * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
//    * padding value(preset in python).
//    */
//   for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
//     int32_t expert_id = topk_ids[i];
//     /** The cumsum[expert_id] stores the starting index of the tokens that the
//      * expert with expert_id needs to process, and
//      * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
//      * processed by the expert with expert_id within the current thread's token
//      * shard.
//      */
//     int32_t rank_post_pad =
//         tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
//         cumsum[expert_id];
//     sorted_token_ids[rank_post_pad] = i;
//     ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
//   }
// }
// }  // namespace sarathi

// void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
//                           int64_t block_size, torch::Tensor sorted_token_ids,
//                           torch::Tensor experts_ids,
//                           torch::Tensor num_tokens_post_pad) {
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   SARATHI_DISPATCH_INTEGRAL_TYPES(
//       topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
//         // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
//         // tensors
//         const int32_t shared_mem =
//             ((num_experts + 1) * num_experts + (num_experts + 1)) *
//             sizeof(int32_t);

//         // set dynamic shared mem
//         auto kernel = sarathi::moe_align_block_size_kernel<scalar_t>;
//         AT_CUDA_CHECK(SARATHI_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
//             (void*)kernel, shared_mem));
//         kernel<<<1, num_experts, shared_mem, stream>>>(
//             topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
//             experts_ids.data_ptr<int32_t>(),
//             num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
//             topk_ids.numel());
//       });
// }


template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded) {
  extern __shared__ int32_t shared_counts[];

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    *total_tokens_post_pad = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with 0
  const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x) {
    expert_ids[i] = 0;
  }
}


template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel, int32_t num_experts) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  SARATHI_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
        // tensors
        const int32_t shared_mem =
            ((num_experts + 1) * num_experts + (num_experts + 1)) *
            sizeof(int32_t);
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::empty({num_experts + 1}, options_int);

        // set dynamic shared mem
        auto align_kernel = moe_align_block_size_kernel<scalar_t>;
        AT_CUDA_CHECK(SARATHI_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
            (void*)align_kernel, shared_mem));

        size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
        size_t shared_mem_size =
            num_warps * experts_per_warp * sizeof(int32_t);

        align_kernel<<<1, threads, shared_mem_size, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
            padded_num_experts, experts_per_warp, block_size,
            topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
            sorted_token_ids.size(0));

        const int block_threads = std::min(256, (int)threads);
        const int num_blocks =
            (topk_ids.numel() + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);

        auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
        sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel(), num_experts);
      });
}
