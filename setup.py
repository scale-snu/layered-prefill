import os
from setuptools import setup, find_packages
from typing import List

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


this_dir = os.path.dirname(__file__)

ext_modules = []
CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# ccache
if os.system("which ccache > /dev/null 2>&1") == 0 and os.environ.get("CONDA_PREFIX") is not None:
    print("Using ccache for compilation")
    if os.system("which gcc-ccache > /dev/null 2>&1") != 0:
        if os.system(r"""echo "ccache gcc \"\$@\"" > $CONDA_PREFIX/bin/gcc-ccache""") != 0:
            raise RuntimeError("Failed to create gcc-ccache wrapper script")
    if os.system("chmod +x $CONDA_PREFIX/bin/gcc-ccache") != 0:
        raise RuntimeError("Failed to make gcc-ccache wrapper script executable")
    if os.system("which g++-ccache > /dev/null 2>&1") != 0:
        if os.system(r"""echo "ccache g++ \"\$@\"" > $CONDA_PREFIX/bin/g++-ccache""") != 0:
            raise RuntimeError("Failed to create g++-ccache wrapper script")
    if os.system("chmod +x $CONDA_PREFIX/bin/g++-ccache") != 0:
        raise RuntimeError("Failed to make g++-ccache wrapper script executable")
    if os.system("which nvcc-ccache > /dev/null 2>&1") != 0:
        if os.system(r"""echo "ccache nvcc \"\$@\"" > $CONDA_PREFIX/bin/nvcc-ccache""") != 0:
            raise RuntimeError("Failed to create nvcc-ccache wrapper script")
    if os.system("chmod +x $CONDA_PREFIX/bin/nvcc-ccache") != 0:
        raise RuntimeError("Failed to make nvcc-ccache wrapper script executable")

    # os.environ["CC"] = "gcc-ccache"
    # os.environ["CXX"] = "g++-ccache"
    os.environ["PYTORCH_NVCC"] = "nvcc-ccache"

ops_extension = CUDAExtension(
    name="nanovllm.ops",
    sources=[
        "csrc/ops.cpp",
        "csrc/pos_encoding_kernels.cu",
        "csrc/layernorm_kernels.cu",
        "csrc/activation_kernels.cu",
        "csrc/moe_align_block_size_kernels.cu",
        "csrc/moe_topk_softmax_kernels.cu",
        "csrc/store_kvcache_kernels.cu",
        "csrc/custom_all_reduce_kernels.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    extra_link_args=['-Wl,--no-as-needed', '-lcuda'],
)
ext_modules.append(ops_extension)

# # Positional encoding kernels.
# positional_encoding_extension = CUDAExtension(
#     name="nanovllm.pos_encoding_ops",
#     sources=["csrc/pos_encoding.cpp", "csrc/pos_encoding_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
# )
# ext_modules.append(positional_encoding_extension)

# # Layer normalization kernels.
# layernorm_extension = CUDAExtension(
#     name="nanovllm.layernorm_ops",
#     sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
# )
# ext_modules.append(layernorm_extension)

# # Activation kernels.
# activation_extension = CUDAExtension(
#     name="nanovllm.activation_ops",
#     sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
# )
# ext_modules.append(activation_extension)

# # Fused MoE kernels.
# moe_extension = CUDAExtension(
#     name="nanovllm.moe_ops",
#     sources=["csrc/moe.cpp", "csrc/moe_align_block_size_kernels.cu", "csrc/moe_topk_softmax_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
# )
# ext_modules.append(moe_extension)

# # Store KV cache kernels.
# store_kvcache_extension = CUDAExtension(
#     name="nanovllm.store_kvcache_ops",
#     sources=["csrc/store_kvcache.cpp", "csrc/store_kvcache_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
# )
# ext_modules.append(store_kvcache_extension)

# # Custom all-reduce kernels.
# custom_all_reduce_extension = CUDAExtension(
#     name="nanovllm.custom_all_reduce_ops",
#     sources=["csrc/custom_all_reduce.cpp", "csrc/custom_all_reduce_kernels.cu"],
#     extra_compile_args={
#         "cxx": CXX_FLAGS,
#         "nvcc": NVCC_FLAGS,
#     },
#     extra_link_args=['-Wl,--no-as-needed', '-lcuda'],
# )
# ext_modules.append(custom_all_reduce_extension)


setup(
    name="layered-prefill",
    version="0.1.0",
    packages=find_packages(),
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
)
