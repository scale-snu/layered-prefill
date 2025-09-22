# Staged Prefill

## How to install

```bash
conda install -n layered-prefill cuda-toolkit cuda-version=12.8 cmake ninja ccache c-compiler cxx-compiler -c nvidia
pip install torch uv flash-attn
pip install -e . --no-build-isolation
CCACHE_NOHASHDIR="true" uv pip install -e flash-attention --verbose --refresh --no-build-isolation
```

## How to run

```bash
CUDA_VISIBLE_DEVICES=1 python nanovllm/entrypoints/api_server.py --model /data/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5/ --max-num-batched-tokens 256 --max-num-seqs 16 --max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --schedule-mode chunked-prefill --num-stages 4
```

Qwen/Qwen3-30B-A3B 16384 16384 1394.94 8192 1479.21 4096 1563.45 2048 1782.23 1024 2625.53 512 4413.58
