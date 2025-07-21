import os
import time
import psutil
import subprocess
import itertools

import json
import uuid
from typing import AsyncGenerator

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.config import APIServerConfig

GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


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
            "/data/cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9/",
        ],
        # "max_num_batched_tokens": [256, 512, 1024],
        "max_num_batched_tokens": [512],
        "max_num_seqs": [16],
        "max_model_len": [16384],
        "gpu_memory_utilization": [0.9],
        "tensor_parallel_size": [1],
        "enforce_eager": [False],
        "log_level": ["debug"],
        "host": ["localhost"],
        "port": [8001],
        "nccl_port": [2334],
        "schedule_mode": ["staged-prefill", "chunked-prefill"],
        "num_stages": [2, 4],
    }
    request_configs = {
        "input_lengths": [1024, 16384],
        "output_lengths": [128],
        "request_rate": [0.1, 0.125, 0.15, 0.175, 0.2, 1, 1.2, 1.3, 1.5, 1.7, 1.8, 2, 2.5, 3],
        "num_requests": [300],
    }
    MAX_TIME = 300

    for model, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, tensor_parallel_size, enforce_eager, log_level, host, port, nccl_port, schedule_mode, num_stages in itertools.product(
        server_configs["models"],
        server_configs["max_num_batched_tokens"],
        server_configs["max_num_seqs"],
        server_configs["max_model_len"],
        server_configs["gpu_memory_utilization"],
        server_configs["tensor_parallel_size"],
        server_configs["enforce_eager"],
        server_configs["log_level"],
        server_configs["host"],
        server_configs["port"],
        server_configs["nccl_port"],
        server_configs["schedule_mode"],
        server_configs["num_stages"]
    ):
        if schedule_mode == "chunked-prefill":
            num_stages = 2
        elif schedule_mode == "staged-prefill":
            max_num_batched_tokens = 256
        print(f"Running benchmark with config: {model}, {max_num_batched_tokens}, {max_num_seqs}, {max_model_len}, {gpu_memory_utilization}, {tensor_parallel_size}, {enforce_eager}, {log_level}, {host}, {port}, {nccl_port}, {schedule_mode}, {num_stages}")

        server_command = f"python -m nanovllm.entrypoints.api_server --model {model} --max-num-batched-tokens {max_num_batched_tokens} --max-num-seqs {max_num_seqs} --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} --tensor-parallel-size {tensor_parallel_size} --enforce-eager {'--enforce-eager' if enforce_eager else ''} --log-level {log_level} --host {host} --port {port} --nccl-port {nccl_port} --schedule-mode {schedule_mode} --num-stages {num_stages}"

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

        for input_length, output_length, request_rate, num_requests in itertools.product(
            request_configs["input_lengths"],
            request_configs["output_lengths"],
            request_configs["request_rate"],
            request_configs["num_requests"]
        ):
            if schedule_mode == "staged-prefill":
                if num_stages == 2:
                    input_length = 1024
                elif num_stages == 4:
                    input_length = 16384

            if input_length == 16384:
                if request_rate > 0.2:
                    request_rate = 0.2
            elif input_length == 1024:
                num_stages = 2
                if request_rate < 1:
                    request_rate = 1
            num_requests = min(num_requests, int(MAX_TIME / (1 / request_rate)))
            print(f"Running benchmark with request config: {input_length}, {output_length}, {request_rate}, {num_requests}")

            log_filename = f"logs/benchmark_{model.split('/')[-4]}_{max_num_batched_tokens}_{max_num_seqs}_{max_model_len}_{gpu_memory_utilization}_{tensor_parallel_size}_{enforce_eager}_{log_level}_{host}_{port}_{nccl_port}_{schedule_mode}_{num_stages}_{input_length}_{output_length}_{request_rate}_{num_requests}.log"

            if os.path.exists(log_filename):
                print(f"Log file {log_filename} already exists. Skipping this configuration.")
                continue

            # Run the benchmark command
            benchmark_command = f"python benchmarks/benchmark_serving.py --model {model} --endpoint /generate --dataset-name random --request-rate {request_rate} --percentile-metrics 'ttft,tpot,itl,e2el' --metric-percentiles '50,90,95,99' --goodput 'ttft:200' 'tpot:20' 'e2el:20000' --num-prompts {num_requests} --random-input-len {input_length} --random-output-len {output_length} --port {port} --backend nano-vllm --ignore-eos > {log_filename} 2>&1"
            benchmark_process = run_command(benchmark_command)

            # Wait for the benchmark to finish
            benchmark_process.wait()

            print(f"Benchmark ({input_length}, {output_length}, {request_rate}, {num_requests}) completed with return code: {benchmark_process.returncode}")

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
