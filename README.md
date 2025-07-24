# Staged Prefill

## How to install

```bash
pip install torch
pip install -e .
```

## How to run

```bash
CUDA_VISIBLE_DEVICES=1 python nanovllm/entrypoints/api_server.py --model /data/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5/ --max-num-batched-tokens 256 --max-num-seqs 16 --max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --schedule-mode chunked-prefill --num-stages 4
```
