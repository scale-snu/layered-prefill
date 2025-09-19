import os
import time
import psutil
import socket
import subprocess
import itertools

import json
import uuid
from typing import AsyncGenerator

import torch
from zeus.monitor import ZeusMonitor

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.config import APIServerConfig

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
GPU_IDS = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
ENV = r"""TORCH_CUDA_ARCH_LIST="8.0;9.0" PATH=$PATH:{conda_prefix}/nvvm/bin CUDA_HOME={conda_prefix}/targets/x86_64-linux CUDA_VISIBLE_DEVICES={gpu_id}""".format(conda_prefix=CONDA_PREFIX, gpu_id=",".join(GPU_IDS))
GPU_ID = int(GPU_IDS[0])

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_gpu_memory(gpu_id):
    try:
        result = subprocess.run(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader --id={gpu_id}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Failed to check GPU memory: {gpu_id}")
            return False

        lines = result.stdout.decode("utf-8").strip().split("\n")
        if len(lines) > 1:
            print(f"Failed to check GPU memory: {gpu_id}")
            print(lines)
            return False

        memory_line = lines[0]

        memory_used = float(memory_line.replace("MiB", ""))
        if memory_used > 100:
            print(f"GPU memory is not released: {gpu_id} ({memory_used} MiB)")
            return False

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to check GPU memory: {e}")
        return False


def wait_for_gpu_memory_release(gpu_id):
    is_gpu_memory_released = check_gpu_memory(gpu_id)
    while not is_gpu_memory_released:
        print(f"Waiting for GPU memory to be released: {gpu_id}")
        time.sleep(10)
        is_gpu_memory_released = check_gpu_memory(gpu_id)


def run_command(command):
    try:
        print(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True)
        return process
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        raise


if __name__ == "__main__":
    server_configs = {
        "models": [
            # "/data/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5/",
            # "/data/cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9/",
            # "/data/cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb/",

            # "/home/gunjunlee/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/",
            # "/home/gunjunlee/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/",
            "/home/gunjunlee/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/",
        ],
        # "max_num_batched_tokens": [128, 256, 512, 1024, 2048, 16384],
        # "max_num_batched_tokens": [4096],
        # "max_num_batched_tokens": [1024],
        # "max_num_batched_tokens": [256],
        # "max_num_batched_tokens": [16384],
        # "max_num_batched_tokens": [2048],
        "max_num_batched_tokens": [512],
        # "max_num_batched_tokens": [8192],
        # "max_num_batched_tokens": [6144],
        "max_num_seqs": [256],
        # "max_num_seqs": [128],
        "max_model_len": [32768],
        # "gpu_memory_utilization": [0.9],
        "gpu_memory_utilization": [0.85],
        "tensor_parallel_size": [len(GPU_IDS)],
        "enforce_eager": [False],
        "log_level": ["debug"],
        "host": ["localhost"],
        # "schedule_mode": ["staged-prefill", "chunked-prefill"],
        # "schedule_mode": ["staged-prefill"],
        "schedule_mode": ["chunked-prefill"],
        # "num_stages": [1, 2, 4, 8, 16],
        # "num_stages": [1, 2, 4, 5, 6, 7, 8, 9, 12],
        # "num_stages": [5, 6, 7, 8, 9],
        "num_stages": [1],
        # "num_stages": [4],
        # "num_stages": [8],
        # "num_stages": [12],
        # "num_stages": [24],
        # "num_stages": [16],
    }
    request_configs = {
        # "datasets": [("random", 1024, 128), ("random", 16384, 128), ("sharegpt", -1, None), ("arxiv", -1, None), ("longbench", -1, 8192)],
        # "datasets": [("sharegpt", -1, None)],
        "datasets": [("arxiv", -1, None)],
        # "datasets": [("longbench", -1, 8192)],
        # A100
        # "request_rate": [1, 1.1, 1.2, 1.3, 1.4, 1.5],  # Qwen Moe sharegpt request rate
        # "request_rate": [1, 1.25, 1.5, 1.75, 2.0],  # GPT Moe sharegpt request rate
        # "request_rate": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50],  # Qwen Moe arxiv request rate
        # "request_rate": [0.4, 0.5, 0.6, 0.7, 0.8],  # GPT Moe arxiv request rate
        # "request_rate": [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3],  # Qwen Moe longbench request rate
        # "request_rate": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],  # GPT Moe longbench request rate
        # H100
        # "request_rate": [1, 1.2, 1.4, 1.6, 1.8, 2.0],  # Qwen Moe sharegpt request rate
        # "request_rate": [1, 1.5, 2.0, 2.5, 3.0],  # GPT Moe sharegpt request rate
        # "request_rate": [0.4, 0.5, 0.6, 0.7, 0.8],  # Qwen Moe arxiv request rate
        # "request_rate": [0.7, 0.8, 0.9, 1.0, 1.1],  # GPT Moe arxiv request rate
        # "request_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # Qwen Moe longbench request rate
        # "request_rate": [0.1, 0.15, 0.20, 0.25, 0.30],  # GPT Moe longbench request rate
        # H100x2
        # "request_rate": [1, 1.5, 2.0, 2.5, 3.0],  # Qwen Moe sharegpt request rate
        # "request_rate": [2.0, 2.5, 3.0, 4.0, 5.0],  # GPT Moe sharegpt request rate
        # "request_rate": [0.5, 0.7, 0.9, 1.1, 1.3],  # Qwen Moe arxiv request rate
        "request_rate": [0.5, 0.75, 1.0, 1.25, 1.5],  # GPT Moe arxiv request rate
        # "request_rate": [0.15, 0.30, 0.45, 0.60, 0.75],  # Qwen Moe longbench request rate
        # "request_rate": [0.2, 0.25, 0.3, 0.35, 0.4],  # GPT Moe longbench request rate
        # "request_rate": [1],  # Moe sharegpt ttft-tbt
        # "request_rate": [0.15],  # Moe longbench ttft-tbt
        # "request_rate": [1, 1.5, 2.0, 2.5, 3.0],  # Non-moe sharegpt request rate
        # "request_rate": [1, 1.25, 1.5, 1.75, 2.0],  # Non-moe sharegpt request rate (new)
        # "request_rate": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],  # Non-moe longbench request rate
        # "request_rate": [0.2, 0.25, 0.3, 0.35, 0.4],  # Non-moe longbench request rate
        # "request_rate": [2.0],  # Non-moe sharegpt ttft-tbt
        "num_requests": [6000],
    }
    MAX_TIME = 600
    WARMUP_TIME = 60

    monitor = ZeusMonitor(gpu_indices=[int(i) for i in GPU_IDS])

    for model, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, tensor_parallel_size, enforce_eager, log_level, host, schedule_mode, num_stages in itertools.product(
        server_configs["models"],
        server_configs["max_num_batched_tokens"],
        server_configs["max_num_seqs"],
        server_configs["max_model_len"],
        server_configs["gpu_memory_utilization"],
        server_configs["tensor_parallel_size"],
        server_configs["enforce_eager"],
        server_configs["log_level"],
        server_configs["host"],
        server_configs["schedule_mode"],
        server_configs["num_stages"]
    ):
        if schedule_mode == "chunked-prefill":
            if num_stages != 1:
                continue
            if max_num_batched_tokens == 8192:
                continue
        elif schedule_mode == "staged-prefill":
            if max_num_batched_tokens != 8192:
                continue

        port = find_free_port()
        nccl_port = find_free_port()

        print(f"Running benchmark with config: {model}, {max_num_batched_tokens}, {max_num_seqs}, {max_model_len}, {gpu_memory_utilization}, {tensor_parallel_size}, {enforce_eager}, {log_level}, {host}, {port}, {nccl_port}, {schedule_mode}, {num_stages}")

        server_command = f"{ENV} python -m nanovllm.entrypoints.api_server --model {model} --max-num-batched-tokens {max_num_batched_tokens} --max-num-seqs {max_num_seqs} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} --tensor-parallel-size {tensor_parallel_size} {'--enforce-eager' if enforce_eager else ''} --log-level {log_level} --host {host} --port {port} --nccl-port {nccl_port} --schedule-mode {schedule_mode} --num-stages {num_stages}"

        server_process = run_command(server_command)
        print(f"Server process started with PID: {server_process.pid}")
        test_command = f"http --stream POST {host}:{port}/generate model=\"\" prompt:='\"hi there\"' max_tokens:=1 temperature:=0.0 stream:=true"
        while True:
            test_process = run_command(test_command)
            test_process.wait()
            if test_process.returncode == 0:
                print("Server is ready.")
                break
            time.sleep(5)
            if server_process.poll() is not None:
                print("Server process terminated unexpectedly.")
                raise RuntimeError("Server process terminated unexpectedly.")

        for datasets, request_rate, num_requests in itertools.product(
            request_configs["datasets"],
            request_configs["request_rate"],
            request_configs["num_requests"]
        ):
            dataset_name, input_length, output_length = datasets
            # if dataset_name == "random":
            #     if input_length == 16384:
            #         if request_rate >= 1:
            #             continue  # Skip high request rates for long inputs
            #     elif input_length == 1024:
            #         if request_rate < 1:
            #             continue  # Skip low request rates for short inputs
            # elif dataset_name == "longbench":
            #     if request_rate > 0.5:
            #         continue
            #     pass
            # elif dataset_name == "arxiv":
            #     if request_rate >= 1 or request_rate < 0.3:
            #         continue
            #     pass
            # elif dataset_name == "sharegpt":
            #     if request_rate < 0.5:
            #         continue
            # else:
            #     raise ValueError(f"Unknown dataset: {dataset_name}")

            num_requests = min(num_requests, int(MAX_TIME / (1 / request_rate)))
            print(f"Running benchmark with request config: {input_length}, {output_length}, {request_rate}, {num_requests}")

            log_filename = f"logs/benchmark_{model.split('/')[-4]}_{max_num_batched_tokens}_{max_num_seqs}_{max_model_len}_{gpu_memory_utilization}_{tensor_parallel_size}_{enforce_eager}_{log_level}_{schedule_mode}_{num_stages}_{dataset_name}_{input_length}_{output_length}_{request_rate}_{num_requests}.log"
            json_filename = log_filename.replace(".log", ".json")

            os.makedirs(os.path.dirname(json_filename), exist_ok=True)

            if os.path.exists(json_filename):
                print(f"Log file {json_filename} already exists. Skipping this configuration.")
                continue

            warmup_num_requests = min(num_requests, int(WARMUP_TIME / (1 / request_rate)))
            print(f"Running warmup with request config: {input_length}, {output_length}, {request_rate}, {warmup_num_requests}")

            if dataset_name == "random":
                dataset_flag = f"--dataset-name random --random-input-len {input_length} --random-output-len {output_length} --ignore-eos"
            elif dataset_name == "sharegpt":
                dataset_flag = f"--dataset-name {dataset_name}"
                if output_length is not None:
                    dataset_flag += f" --sharegpt-output-len {output_length}"
            elif dataset_name == "arxiv":
                dataset_flag = f"--dataset-name {dataset_name}"
                if output_length is not None:
                    dataset_flag += f" --arxiv-output-len {output_length}"
            elif dataset_name == "longbench":
                dataset_flag = f"--dataset-name {dataset_name} --longbench-output-len {output_length}"
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            warmup_command = f"python benchmarks/benchmark_serving.py --model {model} --endpoint /generate --request-rate {request_rate} --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '50,90,95,99' --goodput 'ttft:200' 'tpot:20' 'e2el:20000' --num-prompts {warmup_num_requests} {dataset_flag} --port {port} --backend nano-vllm > /dev/null 2>&1"

            monitor.begin_window("warmup")
            warmup_process = run_command(warmup_command)

            # Wait for the warmup to finish
            warmup_process.wait()

            result = monitor.end_window("warmup")

            print(f"Warmup completed with return code: {warmup_process.returncode}")
            print(f"Warmup GPU stats: {result.time}s, {result.total_energy} J")

            # Run the benchmark command
            benchmark_command = f"python benchmarks/benchmark_serving.py --model {model} --endpoint /generate --request-rate {request_rate} --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '50,90,95,99' --goodput 'ttft:200' 'tpot:20' 'e2el:20000' --num-prompts {num_requests} {dataset_flag} --port {port} --backend nano-vllm --save-result --save-detailed --result-filename {json_filename} > {log_filename} 2>&1"

            monitor.begin_window("benchmark")
            benchmark_process = run_command(benchmark_command)

            # Wait for the benchmark to finish
            benchmark_process.wait()

            result = monitor.end_window("benchmark")

            print(f"Benchmark ({dataset_name}, {input_length}, {output_length}, {request_rate}, {num_requests}) completed with return code: {benchmark_process.returncode}")
            print(f"Benchmark GPU stats: {result.time}s, {result.total_energy} J")
            with open(log_filename, "a") as f:
                f.write(f"\nGPU stats: {result.time}s, {result.total_energy} J, average power: {result.total_energy / result.time if result.time > 0 else 0} W\n")

        # Terminate the server process
        while not check_gpu_memory(GPU_ID):
            try:
                processes = psutil.Process(server_process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                print("Server process has already terminated.")
                break
            for p in processes:
                print(f"Terminating child process: {p.pid}")
                p.terminate()
            print(f"Terminating server process: {server_process.pid}")
            server_process.terminate()
            time.sleep(5)

        print("GPU memory is released.")
        # Check GPU memory release
        wait_for_gpu_memory_release(GPU_ID)
