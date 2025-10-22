# Layered Prefill

Layered Prefill changes the scheduling axis from tokens to layers and removes redundant MoE weight reloads while keeping decode stall free. The result is lower TTFT, lower end-to-end latency, and lower energy per token without hurting TBT stability.

## How to install

```bash
conda create -n layered-prefill python=3.10 -y
conda install -n layered-prefill cuda-toolkit cuda-version=12.8 cmake ninja ccache c-compiler cxx-compiler -c nvidia
conda activate layered-prefill
pip install torch==2.8.0 uv httpie psutil amd-quark
git clone https://github.com/vllm-project/flash-attention.git flash-attention
cd flash-attention; git checkout d9e577e; patch -p0 < ../flash-attention.patch; cd ..
TORCH_CUDA_ARCH_LIST="8.0;9.0" CCACHE_NOHASHDIR="true" uv pip install -e flash-attention --verbose --refresh --no-build-isolation

CCACHE_NOHASHDIR="true" uv pip install -e . --no-build-isolation --verbose --refresh
```

## How to run

```bash
# chunked prefill
CUDA_VISIBLE_DEVICES=0,1 python nanovllm/entrypoints/api_server.py --model /home/gunjunlee/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/ --max-num-batched-tokens 512 --max-num-seqs 256 --max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --schedule-mode chunked-prefill --num-stages 1

# layered prefill
CUDA_VISIBLE_DEVICES=0,1 python nanovllm/entrypoints/api_server.py --model /home/gunjunlee/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/ --max-num-batched-tokens 8192 --max-num-seqs 256 --max-model-len 32768 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --schedule-mode layered-prefill --num-stages 16
```

## Algorithm

![Layered Prefill](assets/layered_prefill.png)

The model is partitioned into contiguous layer groups and prefill advances one group per iteration while every group continues to run decode. At each iteration exactly one designated group performs decode with prefill for newly admitted requests. All other groups execute decode only. Prefill then moves to the next group in the following iteration and completes after the number of groups many iterations. Decode never pauses and stall free behavior holds throughout.

The key effect is that a prompt traverses each layer once during prefill. Chunk based methods repeat the full stack for every chunk and reload MoE experts over and over. Layered Prefill eliminates this chunk amplified reload. Off chip bandwidth drops and energy follows. Because decode work exists in every iteration, TBT remains within the SLO envelope.

The method is orthogonal to chunking. When very long inputs must be pipelined, you can still chunk while using Layered Prefill to raise the chunk size safely. Fewer chunks mean fewer expert reloads and less bandwidth. With sufficiently large effective chunks MoE shifts from memory bound toward compute bound which further moderates latency growth.

We made the following key observations. First, Layered Prefill expands the TTFT and TBT Pareto frontier and sustains higher SLO attainment at higher request rates than chunked prefill. Queueing and prefill time drop while TBT quality stays strong.

![Pareto frontier](assets/slo_distribution.png)

Second, throughput under SLO improves on both arXiv and ShareGPT style traces. On arXiv the system holds near perfect SLO to higher request rates where chunked prefill already collapses. On ShareGPT the advantage persists at higher load as well.

![Pareto frontier](assets/slo_attainment.png)

Third, latency quality improves. At the same request rate on arXiv with a representative MoE model the mean TTFT falls by about half and the tail TTFT drops markedly. The token generation trajectory shows an earlier first token and a steeper slope over wall clock time which shortens end-to-end latency for a single request.

Fourth, energy per token goes down. We define energy per token as total GPU energy divided by the sum of prompt and generated tokens. Layered Prefill reduces this metric on both models and datasets. The reduction aligns with the measured cut in redundant expert weight traffic.

Fifth, MoE traffic decreases. Expert weight load bytes shrink on both traces with larger gains for long prompts where chunking would otherwise trigger repeated reloads. The traffic reduction is consistent with the SLO gains at high request rates.

Finally, raising chunk size alone cannot recover the same benefits under chunked prefill. Larger chunks reduce runtime and energy but inflate tail TBT and violate SLO at scale. Layered Prefill preserves the efficiency of large effective chunks without the TBT regressions because decode continues every iteration.

## VS. vLLM (v0.10.2)

Layered prefill shows significant advantages over vLLM's chunked prefill in terms of TTFT, TBT stability and energy efficiency.
We compared both systems using the Qwen3-30B-A3B model on the arXiv trace with identical hardware (a 2x H100 80GB GPU) and similar configurations (tensor parallelism of 2, max model length of 32K tokens, and GPU memory utilization of 0.85).
The results are as follows:

### Overall comparison

| Metric | vLLM | Layered Prefill | Δ (Layered − vLLM) |
|---|---:|---:|---:|
| Mean TTFT (ms) | 1018.75 | 712.84 | **−30.0%** |
| Median TTFT (ms) | 872.98 | 560.54 | **−35.8%** |
| Mean TPOT (ms) | 19.71 | 15.09 | **−23.4%** |
| Median TPOT (ms) | 20.14 | 14.52 | **−27.9%** |
| Mean ITL (ms) | 19.61 | 14.89 | **−24.1%** |
| Median ITL (ms) | 15.42 | 12.74 | **−17.4%** |
| Mean E2E latency (ms) | 4904.02 | 3525.08 | **−28.1%** |
| Median E2E latency (ms) | 4424.63 | 3123.80 | **−29.4%** |

### Latency percentiles

#### TTFT

| Percentile | vLLM (ms) | Layered Prefill (ms) |
|---|---:|---:|
| P5 | 231.98 | 101.10 |
| P10 | 305.23 | 190.88 |
| P50 | 872.98 | 560.54 |
| P90 | 1829.95 | 1430.42 |
| P95 | 2383.19 | 2105.67 |
| P99 | 2812.16 | 2656.01 |
| P99.9 | 3050.27 | 2936.77 |
| P100 | 3076.73 | 2967.97 |

#### Inter-token latency (ITL)

| Percentile | vLLM (ms) | Layered Prefill (ms) |
|---|---:|---:|
| P5 | 7.16 | 7.01 |
| P10 | 8.47 | 7.66 |
| P50 | 15.42 | 12.74 |
| P90 | 32.56 | 25.54 |
| P95 | 34.00 | 26.74 |
| P99 | 36.79 | 27.90 |
| P99.9 | 41.65 | 35.13 |
| P100 | 45.37 | 39.80 |

#### End-to-end latency

| Percentile | vLLM (ms) | Layered Prefill (ms) |
|---|---:|---:|
| P5 | 1674.82 | 1111.56 |
| P10 | 2287.97 | 1349.73 |
| P50 | 4424.63 | 3123.80 |
| P90 | 8332.01 | 6158.27 |
| P95 | 9541.14 | 7022.79 |
| P99 | 10446.70 | 7822.13 |
| P99.9 | 10862.74 | 8197.62 |
| P100 | 10908.97 | 8239.34 |

### Commands to reproduce

```
# Start the API server
Layered-prefill: TORCH_CUDA_ARCH_LIST="8.0;9.0" PATH=$PATH:$CONDA_PREFIX/nvvm/bin CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux CUDA_VISIBLE_DEVICES=0,1 python nanovllm/entrypoints/api_server.py --model /home/gunjunlee/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/ --max-num-batched-tokens 8192 --max-num-seqs 256 --max-model-len 16384 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --log-level debug --host localhost --port 8000 --nccl-port 51981 --schedule-mode layered-prefill --num-stages 16
vLLM: CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-30B-A3B --no-enable-prefix-caching --tensor-parallel-size 2 --max-num_batched-tokens 512 --max-model-len 16384 --max-num-seqs 256 --gpu-memory-utilization 0.85

# Run the benchmark
Layered-prefill: python benchmarks/benchmark_serving.py --model Qwen/Qwen3-30B-A3B --endpoint /generate --request-rate 1.5 --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '5,10,50,90,95,99,99.9,100' --goodput 'ttft:200' 'tpot:20' 'e2el:20000' --num-prompts 100 --dataset-name arxiv --port 8000 --backend nano-vllm
vLLM: python benchmarks/benchmark_serving.py --model Qwen/Qwen3-30B-A3B --endpoint /v1/completions --request-rate 1.5 --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '5,10,50,90,95,99,99.9,100' --goodput 'ttft:200' 'tpot:20' 'e2el:20000' --num-prompts 100 --dataset-name arxiv --port 8000 --backend vllm
```

## Citation

If you use layered prefill for your research, please cite our [paper](https://arxiv.org/abs/2510.08055):
```bibtex
@misc{lee2025tokenslayersredefiningstallfree,
      title={From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill},
      author={Gunjun Lee and Jiwon Kim and Jaiyoung Park and Younjoo Lee and Jung Ho Ahn},
      year={2025},
      eprint={2510.08055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.08055},
}
```
