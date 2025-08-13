# Staged Prefill

## How to install

```bash
conda create -n layered-prefill python=3.10 -y
conda install -n layered-prefill cuda-toolkit cuda-version=12.8 cmake ninja ccache c-compiler cxx-compiler -c nvidia
conda activate layered-prefill
pip install torch==2.8.0 uv httpie psutil amd-quark
# pip install flash-attn
CCACHE_NOHASHDIR="true" uv pip install -e . --no-build-isolation --verbose --refresh
TORCH_CUDA_ARCH_LIST="8.0;9.0" CCACHE_NOHASHDIR="true" uv pip install -e flash-attention --verbose --refresh --no-build-isolation
```

## How to run

```bash
CUDA_VISIBLE_DEVICES=1 python nanovllm/entrypoints/api_server.py --model /data/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5/ --max-num-batched-tokens 256 --max-num-seqs 16 --max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --schedule-mode chunked-prefill --num-stages 4
```
