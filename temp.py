import os
import json
import datetime
import shutil
from queue import PriorityQueue

import numpy as np
import pandas as pd
from tqdm import tqdm

# json_filename_format = f"logs/benchmark_{model.split('/')[-4]}_{max_num_batched_tokens}_{max_num_seqs}_{max_model_len}_{gpu_memory_utilization}_{tensor_parallel_size}_{enforce_eager}_{log_level}_{host}_{port}_{nccl_port}_{schedule_mode}_{num_stages}_{input_length}_{output_length}_{request_rate}_{num_requests}.log"


datas = []

slo_constraints = {
    "models--Qwen--Qwen3-30B-A3B": {
        "longbench": {
            "ttft": 10,
            # "itl": None,
            "itl": 0.20,
            "tpot": 0.20,
            # "tpot": 0.1,
            # "tpot": None,
        },
        "arxiv": {
            "ttft": 10,
            # "itl": None,
            "itl": 0.20,
            "tpot": 0.20,
            # "tpot": 0.1,
            # "tpot": None,
        },
        "random": {
            "ttft": 2,
            "itl": 0.20,
            "tpot": 0.20,
        },
        "sharegpt": {
            "ttft": 5,
            # "itl": None,
            "itl": 0.20,
            "tpot": 0.20,
            # "tpot": 0.1,
            # "tpot": None,
        },
    },
    "models--openai--gpt-oss-20b": {
        "longbench": {
            "ttft": 10,
            # "itl": None,
            "itl": 0.18,
            # "tpot": 0.18,
            # "tpot": 0.1,
            "tpot": None,
        },
        "arxiv": {
            "ttft": 10,
            # "itl": None,
            "itl": 0.18,
            # "tpot": 0.18,
            # "tpot": 0.1,
            "tpot": None,
        },
        "random": {
            "ttft": 2,
            "itl": 0.18,
            # "tpot": 0.18,
            "tpot": None,
        },
        "sharegpt": {
            "ttft": 5,
            # "itl": None,
            "itl": 0.18,
            # "tpot": 0.18,
            # "tpot": 0.1,
            "tpot": None,
        },
    },
    "models--Qwen--Qwen3-8B": {
        "longbench": {
            "ttft": 5,
            # "itl": None,
            "itl": 1.00,
            "tpot": 0.20,
            # "tpot": 0.08,
        },
        "random": {
            "ttft": 2,
            "itl": None,
            "tpot": 0.2,
        },
        "sharegpt": {
            "ttft": 3,
            # "itl": None,
            "itl": 0.20,
            "tpot": None,
            # "tpot": 0.08,
        },
    }
}

log_dir = "logs"
for filename in tqdm(os.listdir(log_dir)):
    if filename.endswith(".json"):
        json_filename = os.path.join(log_dir, filename)
        _, model_name, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, tensor_parallel_size, enforce_eager, log_level, schedule_mode, num_stages, dataset_name, input_length, output_length, request_rate, num_requests = filename[:-len(".json")].split('_')
        # print(model_name, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, tensor_parallel_size, enforce_eager, log_level, schedule_mode, num_stages, dataset_name, input_length, output_length, request_rate, num_requests)

        with open(json_filename, 'r') as f:
            data = json.load(f)
            ttfts = data.get("ttfts", [])
            itls = data.get("itls", [])
            if not ttfts or not itls:
                print(f"Skipping {json_filename} due to missing data.")
                continue

            max_requests = -1
            mean_requests = -1
            if "request_dts" in data:
                pq = PriorityQueue()
                request_dts = data["request_dts"]
                for request_dt, ttft, itl in zip(request_dts, ttfts, itls):
                    start_dt = datetime.datetime.fromisoformat(request_dt)
                    end_dt = start_dt + datetime.timedelta(seconds=ttft + sum(itl))
                    for _itl in itl:
                        if _itl > 0.2 and model_name == "models--openai--gpt-oss-20b" and dataset_name == "sharegpt":
                            pq.put((start_dt + datetime.timedelta(seconds=ttft + _itl), 100 + _itl))
                        else:
                            pq.put((start_dt + datetime.timedelta(seconds=ttft + _itl), 0))
                    pq.put((start_dt, 1))
                    pq.put((end_dt, -1))

                current_requests = 0
                max_requests = 0
                _num_requests = []
                while not pq.empty():
                    date, change = pq.get()
                    if model_name == "models--openai--gpt-oss-20b" and dataset_name == "sharegpt":
                        print(date)
                    current_requests += change
                    if change > 0:
                        if change > 100:
                            import pdb; pdb.set_trace()
                        max_requests = max(max_requests, current_requests)
                        _num_requests.append(current_requests)
                mean_requests = np.mean(_num_requests)

            slo = np.ones(len(ttfts), dtype=bool)
            mean_ttft = np.mean(ttfts)
            ttft_slo_attain = 0
            if slo_constraints[model_name][dataset_name]["ttft"] is not None:
                ttft_slo = (np.array(ttfts) <= slo_constraints[model_name][dataset_name]["ttft"])
                slo = slo & ttft_slo
                ttft_slo_attain = np.mean(ttft_slo) * 100

            mean_itl = np.mean(np.concatenate(itls))
            tpots = np.array([np.mean(itl) if len(itl) > 0 else 0 for itl in itls])
            mean_tpot = np.mean(tpots)
            tpot_slo_attain = 0
            if slo_constraints[model_name][dataset_name]["tpot"] is not None:
                tpot_slo = (tpots <= slo_constraints[model_name][dataset_name]["tpot"])
                slo = slo & tpot_slo
                tpot_slo_attain = np.mean(tpot_slo) * 100

            itl_slo_attain = 0
            if slo_constraints[model_name][dataset_name]["itl"] is not None:
                max_itl_len = max(len(itl) for itl in itls)
                itl_slo = (np.stack([np.pad(itl, (0, max_itl_len - len(itl))) for itl in itls], axis=0) <= slo_constraints[model_name][dataset_name]["itl"]).all(axis=1)
                # if model_name == "models--openai--gpt-oss-20b" and dataset_name == "sharegpt":
                #     import pdb; pdb.set_trace()
                slo = slo & itl_slo
                itl_slo_attain = np.mean(itl_slo) * 100

            p50_ttft, p90_ttft, p95_ttft, p99_ttft, p995_ttft = np.percentile(ttfts, [50, 90, 95, 99, 99.5])
            p50_itl, p90_itl, p95_itl, p99_itl, p995_itl = np.percentile(np.concatenate(itls), [50, 90, 95, 99, 99.5])
            p50_tpot, p90_tpot, p95_tpot, p99_tpot, p995_tpot = np.percentile(tpots, [50, 90, 95, 99, 99.5])
            slo_attain = np.mean(slo) * 100

            # print(f"{model_name}, {max_num_batched_tokens}, {max_num_seqs}, {max_model_len}, {gpu_memory_utilization}, {tensor_parallel_size}, {enforce_eager}, {log_level}, {schedule_mode}, {num_stages}, {dataset_name}, {input_length}, {output_length}, {request_rate}, {num_requests}, "
            #       f"{mean_ttft}, {p50_ttft}, {p90_ttft}, {p95_ttft}, {p99_ttft}, "
            #       f"{mean_itl}, {p50_itl}, {p90_itl}, {p95_itl}, {p99_itl}, "
            #       f"{max_requests}, {mean_requests}")

            data_entry = {
                "model_name": model_name,
                "max_num_batched_tokens": int(max_num_batched_tokens),
                "max_num_seqs": int(max_num_seqs),
                "max_model_len": int(max_model_len),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "tensor_parallel_size": int(tensor_parallel_size),
                "enforce_eager": enforce_eager,
                "log_level": log_level,
                "schedule_mode": schedule_mode,
                "num_stages": int(num_stages),
                "dataset_name": dataset_name,
                "input_length": int(input_length),
                "output_length": int(output_length) if output_length.isdigit() else output_length,
                "request_rate": float(request_rate),
                "num_requests": int(num_requests),
                "mean_ttft": mean_ttft,
                "p50_ttft": p50_ttft,
                "p90_ttft": p90_ttft,
                "p95_ttft": p95_ttft,
                "p99_ttft": p99_ttft,
                "p995_ttft": p995_ttft,
                "mean_itl": mean_itl,
                "p50_itl": p50_itl,
                "p90_itl": p90_itl,
                "p95_itl": p95_itl,
                "p99_itl": p99_itl,
                "p995_itl": p995_itl,
                "mean_tpot": mean_tpot,
                "p50_tpot": p50_tpot,
                "p90_tpot": p90_tpot,
                "p95_tpot": p95_tpot,
                "p99_tpot": p99_tpot,
                "p995_tpot": p995_tpot,
                "max_requests": max_requests,
                "mean_requests": mean_requests,
                "slo": slo_attain,
                "ttft_slo_attain": ttft_slo_attain,
                "tpot_slo_attain": tpot_slo_attain,
                "itl_slo_attain": itl_slo_attain,
            }
            datas.append(data_entry)

df = pd.DataFrame(datas)
df = df.sort_values(by=["model_name", "schedule_mode", "max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "enforce_eager", "log_level", "num_stages", "dataset_name", "input_length", "output_length", "request_rate", "num_requests"])
df.to_csv("benchmark_results.csv", index=False)
print("Benchmark results saved to benchmark_results.csv")
