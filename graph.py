import json
import os
import datetime
import shutil
from queue import PriorityQueue

import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.family'] = 'Arial'


def query(cur, sql):
    exe = cur.execute(sql)
    column_names = [description[0] for description in exe.description]
    return pd.DataFrame(exe.fetchall(), columns=column_names)


def connect(path):
    cur = duckdb.connect()
    cur.execute(f"""
        INSTALL sqlite;
        LOAD sqlite;
        ATTACH '{path}' AS report (TYPE sqlite);
        USE report;
        CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start BIGINT,
            "end" BIGINT,
            correlationId BIGINT
        );
    """)
    return cur


def get_kernel_btw(cur, start, end):
    kernel_df = query(cur, f"""
        WITH cupti_kernel AS (
        SELECT
            cupti_kernel.start as start_time,
            cupti_kernel.end as end_time,
            cupti_kernel.end - cupti_kernel.start as duration,
            cupti_kernel.correlationId,
            cupti_kernel.deviceId,
            string_ids.value AS cupti_kernel_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS cupti_kernel
        LEFT JOIN StringIds AS string_ids
            ON string_ids.id = cupti_kernel.demangledName
        )
        -- , graph_kernel AS (
        -- SELECT
        --     cupti_graph_trace.start as start_time,
        --     cupti_graph_trace.end as end_time,
        --     cupti_graph_trace.end - cupti_graph_trace.start as duration,
        --     cupti_graph_trace.correlationId,
        --     'GRAPH' AS cupti_kernel_name
        --     FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE as cupti_graph_trace
        -- )
        -- , total_kernel AS (
        --     SELECT
        --         *
        --     FROM cupti_kernel
        --     UNION ALL
        --     SELECT
        --         *
        --     FROM graph_kernel
        -- )
        , kernel AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY cupti_kernel.end_time) row_num,
                cupti_kernel.start_time as start_time,
                cupti_kernel.end_time as end_time,
                cupti_kernel.duration as duration,
                cupti_kernel.cupti_kernel_name AS cupti_kernel_name,
                cupti_kernel.deviceId AS device_id
            FROM cupti_kernel AS cupti_kernel
            WHERE 1=1
                AND cupti_kernel.start_time >= {start}
                AND cupti_kernel.end_time <= {end}
                AND cupti_kernel.deviceId = 1
        )
        SELECT *
        FROM kernel
        ORDER BY kernel.start_time
    """)
    return kernel_df


# -----------------------------
# Added: UX + Fairness metrics
# -----------------------------
def jain_index(x: np.ndarray, eps: float = 1e-12) -> float:
    """Jain's fairness index. Higher is more fair (max=1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    s1 = np.sum(x)
    s2 = np.sum(x * x)
    if s2 < eps:
        return np.nan
    return (s1 * s1) / (x.size * s2 + eps)


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))


def compute_streaming_ux_fairness_metrics(
    ttfts: list,
    itls: list,
    itl_slo: float | None = None,
    stall_thresholds_s=(0.2, 0.5),
):
    """
    Compute additional metrics for streaming UX and fairness.

    UX (smoothness / burstiness)
    - ITL std, ITL CV
    - ITL burst ratio (p95/p50)
    - Stall rates (fraction of tokens whose ITL > threshold)
    - Per-request mean ITL distribution (p99 over requests)

    Fairness
    - Jain index over per-request *completion rate* = 1 / (TTFT + sum(ITL))
    - Jain index over per-request *decode token throughput* = out_tokens / sum(ITL)
        (only meaningful when output length varies)
    """
    ttfts_np = np.asarray(ttfts, dtype=np.float64)

    # Flatten all token gaps (ITL)
    all_itl = np.concatenate(itls) if len(itls) > 0 else np.array([], dtype=np.float64)

    # Basic ITL stats (smoothness)
    mean_itl = float(np.mean(all_itl)) if all_itl.size > 0 else np.nan
    std_itl = float(np.std(all_itl)) if all_itl.size > 0 else np.nan
    cv_itl = float(std_itl / (mean_itl + 1e-12)) if all_itl.size > 0 else np.nan

    p50_itl = safe_percentile(all_itl, 50)
    p95_itl = safe_percentile(all_itl, 95)
    p99_itl = safe_percentile(all_itl, 99)

    # Burstiness proxy: how much tail exceeds median
    burst_ratio_p95_p50 = (p95_itl / (p50_itl + 1e-12)) if np.isfinite(p50_itl) else np.nan
    burst_ratio_p99_p50 = (p99_itl / (p50_itl + 1e-12)) if np.isfinite(p50_itl) else np.nan

    # Stall rates (token-level)
    stall_rates = {}
    for thr in stall_thresholds_s:
        key = f"stall_rate_itl_gt_{int(thr*1000)}ms"
        stall_rates[key] = float(np.mean(all_itl > thr) * 100.0) if all_itl.size > 0 else np.nan

    # SLO-relative stall rate (e.g., tokens exceeding 2x ITL SLO)
    slo_relative = {}
    if itl_slo is not None and itl_slo > 0:
        slo_relative["stall_rate_itl_gt_2x_slo"] = float(np.mean(all_itl > (2.0 * itl_slo)) * 100.0) if all_itl.size > 0 else np.nan
        slo_relative["stall_rate_itl_gt_4x_slo"] = float(np.mean(all_itl > (4.0 * itl_slo)) * 100.0) if all_itl.size > 0 else np.nan
    else:
        slo_relative["stall_rate_itl_gt_2x_slo"] = np.nan
        slo_relative["stall_rate_itl_gt_4x_slo"] = np.nan

    # Request-level ITL summary: mean ITL per request and its tail
    req_mean_itl = np.array([np.mean(x) if len(x) > 0 else 0.0 for x in itls], dtype=np.float64)
    req_p99_mean_itl = safe_percentile(req_mean_itl, 99)
    req_p95_mean_itl = safe_percentile(req_mean_itl, 95)

    # Completion times per request (user-perceived total time)
    req_decode_time = np.array([np.sum(x) for x in itls], dtype=np.float64)
    req_total_time = ttfts_np + req_decode_time

    # Fairness: completion rate (faster completion => higher rate)
    completion_rate = 1.0 / (req_total_time + 1e-12)
    fairness_jain_completion_rate = jain_index(completion_rate)

    # Fairness: decode token throughput per request (tokens/sec during decode)
    # If output length fixed, this is less informative but still valid.
    out_tokens = np.array([len(x) for x in itls], dtype=np.float64)
    decode_tput = out_tokens / (req_decode_time + 1e-12)
    fairness_jain_decode_tput = jain_index(decode_tput)

    # Tail user experience metrics at request level
    req_p99_total_time = safe_percentile(req_total_time, 99)
    req_p95_total_time = safe_percentile(req_total_time, 95)

    return {
        # UX: token-level smoothness
        "std_itl": std_itl,
        "cv_itl": cv_itl,
        "burst_ratio_p95_p50_itl": burst_ratio_p95_p50,
        "burst_ratio_p99_p50_itl": burst_ratio_p99_p50,
        "token_p99_itl": p99_itl,
        # UX: request-level smoothness
        "req_p95_mean_itl": req_p95_mean_itl,
        "req_p99_mean_itl": req_p99_mean_itl,
        "req_p95_total_time": req_p95_total_time,
        "req_p99_total_time": req_p99_total_time,
        # Stall rates
        **stall_rates,
        **slo_relative,
        # Fairness
        "jain_completion_rate": fairness_jain_completion_rate,
        "jain_decode_tput": fairness_jain_decode_tput,
    }


if __name__ == "__main__":
    datas = []

    # H100
    slo_constraints = {
        "qwen": {
            "longbench": {"ttft": 10, "itl": 0.125, "tpot": None},
            "arxiv": {"ttft": 10, "itl": 0.125, "tpot": None},
            "random": {"ttft": 2, "itl": 0.125, "tpot": 0.125},
            "sharegpt": {"ttft": 5, "itl": 0.125, "tpot": None},
        },
        "gpt": {
            "longbench": {"ttft": 10, "itl": 0.10, "tpot": None},
            "arxiv": {"ttft": 10, "itl": 0.10, "tpot": None},
            "random": {"ttft": 2, "itl": 0.10, "tpot": None},
            "sharegpt": {"ttft": 5, "itl": 0.10, "tpot": None},
        },
        "models--openai--gpt-oss-120b": {
            "longbench": {"ttft": 10, "itl": 0.10, "tpot": None},
            "arxiv": {"ttft": 10, "itl": 0.10, "tpot": None},
            "random": {"ttft": 2, "itl": 0.10, "tpot": None},
            "sharegpt": {"ttft": 5, "itl": 0.10, "tpot": None},
        },
        "models--Qwen--Qwen3-8B": {
            "longbench": {"ttft": 5, "itl": 1.00, "tpot": 0.20},
            "random": {"ttft": 2, "itl": None, "tpot": 0.2},
            "sharegpt": {"ttft": 3, "itl": 0.20, "tpot": None},
        },
    }

    log_dir = "logs"
    for filename in tqdm(os.listdir(log_dir)):
        if filename.endswith(".json"):
            json_filename = os.path.join(log_dir, filename)
            (
                _,
                model_name,
                max_num_batched_tokens,
                max_num_seqs,
                max_model_len,
                gpu_memory_utilization,
                tensor_parallel_size,
                enforce_eager,
                log_level,
                schedule_mode,
                num_stages,
                dataset_name,
                input_length,
                output_length,
                request_rate,
                num_requests,
            ) = filename[:-len(".json")].split("_")

            with open(json_filename, "r") as f:
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
                        pq.put((start_dt, 1))
                        pq.put((end_dt, -1))

                    current_requests = 0
                    max_requests = 0
                    _num_requests = []
                    while not pq.empty():
                        date, change = pq.get()
                        current_requests += change
                        if change > 0:
                            max_requests = max(max_requests, current_requests)
                            _num_requests.append(current_requests)
                    mean_requests = np.mean(_num_requests)

                slo = np.ones(len(ttfts), dtype=bool)
                mean_ttft = np.mean(ttfts)
                ttft_slo_attain = 0
                if slo_constraints[model_name][dataset_name]["ttft"] is not None:
                    ttft_slo = np.array(ttfts) <= slo_constraints[model_name][dataset_name]["ttft"]
                    slo = slo & ttft_slo
                    ttft_slo_attain = np.mean(ttft_slo) * 100

                mean_itl = np.mean(np.concatenate(itls))
                tpots = np.array([np.mean(itl) if len(itl) > 0 else 0 for itl in itls])
                mean_tpot = np.mean(tpots)
                tpot_slo_attain = 0
                if slo_constraints[model_name][dataset_name]["tpot"] is not None:
                    tpot_slo = tpots <= slo_constraints[model_name][dataset_name]["tpot"]
                    slo = slo & tpot_slo
                    tpot_slo_attain = np.mean(tpot_slo) * 100

                itl_slo_attain = 0
                if slo_constraints[model_name][dataset_name]["itl"] is not None:
                    max_itl_len = max(len(itl) for itl in itls)
                    itl_slo = (
                        np.stack(
                            [np.pad(itl, (0, max_itl_len - len(itl))) for itl in itls],
                            axis=0,
                        )
                        <= slo_constraints[model_name][dataset_name]["itl"]
                    ).all(axis=1)
                    slo = slo & itl_slo
                    itl_slo_attain = np.mean(itl_slo) * 100

                p50_ttft, p90_ttft, p95_ttft, p99_ttft, p995_ttft = np.percentile(
                    ttfts, [50, 90, 95, 99, 99.5]
                )
                p50_itl, p90_itl, p95_itl, p99_itl, p995_itl = np.percentile(
                    np.concatenate(itls), [50, 90, 95, 99, 99.5]
                )
                p50_tpot, p90_tpot, p95_tpot, p99_tpot, p995_tpot = np.percentile(
                    tpots, [50, 90, 95, 99, 99.5]
                )
                var_ttft = np.var(ttfts)
                var_itl = np.var(np.concatenate(itls))
                var_tpot = np.var(tpots)

                slo_attain = np.mean(slo) * 100

                # ------------------------------------------------------
                # Added: compute UX + fairness metrics for this log file
                # ------------------------------------------------------
                itl_slo_value = slo_constraints[model_name][dataset_name]["itl"]
                ux_fair = compute_streaming_ux_fairness_metrics(
                    ttfts=ttfts,
                    itls=itls,
                    itl_slo=itl_slo_value,
                    stall_thresholds_s=(0.2, 0.5),  # you can tune these
                )

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

                    "var_ttft": var_ttft,
                    "var_itl": var_itl,
                    "var_tpot": var_tpot,

                    "max_requests": max_requests,
                    "mean_requests": mean_requests,

                    "slo": slo_attain,
                    "ttft_slo_attain": ttft_slo_attain,
                    "tpot_slo_attain": tpot_slo_attain,
                    "itl_slo_attain": itl_slo_attain,

                    # Added columns: UX + fairness
                    **ux_fair,
                }
                datas.append(data_entry)

    df = pd.DataFrame(datas)
    df = df.sort_values(
        by=[
            "model_name",
            "schedule_mode",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "enforce_eager",
            "log_level",
            "num_stages",
            "dataset_name",
            "input_length",
            "output_length",
            "request_rate",
            "num_requests",
        ]
    )
    df.to_csv("benchmark_results.csv", index=False)
    print("Benchmark results saved to benchmark_results.csv")

    df = pd.read_csv("benchmark_results.csv")

    # print("Model: Qwen3-30B-A3B")
    # data = []
    # for model in ["Qwen3-30B-A3B"]:
    #     for chunk_size in [512, 1024, 2048, 4096, 8192]:
    #         print(chunk_size)
    #         cur = connect(f"logs/{model}-{chunk_size}.sqlite")
    #         kernel_df = get_kernel_btw(cur, 0, 1000000000000)
    #         filtered_kernel_df = kernel_df[kernel_df["start_time"] >= (50e9 if model == "Qwen3-30B-A3B" else 40e9)]
    #         flash_df = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("flash")]
    #         start_time = (flash_df.iloc[len(flash_df) // 3 * 2 - 1]["start_time"] + flash_df.iloc[len(flash_df) // 3 * 2]["start_time"]) / 2
    #         # import pdb; pdb.set_trace()
    #         filtered_kernel_df = filtered_kernel_df[filtered_kernel_df["start_time"] >= start_time].sort_values(by="start_time", ascending=True)
    #         duration = (filtered_kernel_df.iloc[-1]["end_time"] - filtered_kernel_df.iloc[0]["start_time"]) / 1e6
    #         nccl_filter = filtered_kernel_df["cupti_kernel_name"].str.contains("nccl") | filtered_kernel_df["cupti_kernel_name"].str.contains("cross_device_reduce")
    #         print(filtered_kernel_df["cupti_kernel_name"].unique())
    #         nccl_duration = filtered_kernel_df[nccl_filter]["duration"].sum() / 1e6  # Convert to ms
    #         attn_filter = (
    #             filtered_kernel_df["cupti_kernel_name"].str.contains("flash")
    #             & ~filtered_kernel_df["cupti_kernel_name"].str.contains("prepare")
    #             & ~filtered_kernel_df["cupti_kernel_name"].str.contains("Combine")
    #         )
    #         attn_duration = filtered_kernel_df[attn_filter]["duration"].sum() / 1e6  # Convert to ms
    #         moe_filter = filtered_kernel_df["cupti_kernel_name"].str.contains("fused_moe_kernel")
    #         moe_duration = filtered_kernel_df[moe_filter]["duration"].sum() / 1e6  # Convert to ms
    #         nonattn_duration = filtered_kernel_df[~nccl_filter & ~attn_filter & ~moe_filter]["duration"].sum() / 1e6
    #         etc_duration = duration - nccl_duration - attn_duration - moe_duration - nonattn_duration
    #         # print(nonattn_kernel_df.groupby("cupti_kernel_name")["duration"].sum().sort_values(ascending=True).head(10))
    #         print(f"Chunk size: {chunk_size}, Nccl duration: {nccl_duration:.2f}, Attn duration: {attn_duration:.2f} ms, MoE duration: {moe_duration:.2f} ms, Non-attn duration: {nonattn_duration:.2f} ms, Etc duration: {etc_duration:.2f} ms")
    #         total_chunk_num = 8192 // chunk_size
    #         for input_length in [4096, 8192, 16384, 32768]:
    #             if input_length < chunk_size:
    #                 continue
    #             attn_kernel_df = filtered_kernel_df[attn_filter]
    #             attn_kernel_df = attn_kernel_df.sort_values(by="start_time", ascending=True)
    #             chunk_num = input_length // chunk_size
    #             num_attn_call = 48 if model == "Qwen3-30B-A3B" else 24
    #             assert len(attn_kernel_df) == total_chunk_num * num_attn_call, f"Chunk size {chunk_size} and input length {input_length} mismatch: {len(attn_kernel_df)} != {total_chunk_num * num_attn_call}"
    #             attn_kernel_df = attn_kernel_df.iloc[:chunk_num * num_attn_call]
    #             _start_time = attn_kernel_df.iloc[0]["start_time"]
    #             _end_time = attn_kernel_df.iloc[-1]["end_time"]
    #             duration = (_end_time - _start_time) / 1e6  # Convert to ms
    #             moe_kernel_df = filtered_kernel_df[moe_filter & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
    #             nonattn_kernel_df = filtered_kernel_df[~attn_filter & ~moe_filter & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
    #             attn_duration = attn_kernel_df["duration"].sum() / 1e6  # Convert to ms
    #             moe_duration = moe_kernel_df["duration"].sum() / 1e6  # Convert to ms
    #             nonattn_duration = nonattn_kernel_df["duration"].sum() / 1e6
    #             etc_duration = duration - attn_duration - moe_duration - nonattn_duration

    #             data.append({
    #                 "model": model,
    #                 "chunk_size": chunk_size,
    #                 "input_length": input_length,
    #                 "attn_duration": attn_duration,
    #                 "moe_duration": moe_duration,
    #                 "nonattn_duration": nonattn_duration,
    #                 "etc_duration": etc_duration,
    #                 "nccl_duration": nccl_duration,
    #                 "total_duration": duration,
    #             })

    # tdf = pd.DataFrame(data)
    # tdf = tdf[(tdf["chunk_size"] <= 8192) & tdf["input_length"].isin([8192])]

    # # # 8192: 57.1, 4096: 110.1, 2048: 213.3, 1024: 406.3, 512: 764.1
    # # # chunk size, the number of chunks, MoE load Bytes (GB), MoE duration (ms)
    # # # 512, 16, 764.1, 311.95
    # # # 1024, 8, 406.3, 184.88
    # # # 2048, 4, 213.3, 112.28
    # # # 4096, 2, 110.1, 84.57
    # # # 8192, 1, 57.1, 75.68
    # # # chunk size, attn duration, moe duration, etc duration
    # # # 512, 68.975, 311.95, 104.645
    # # # 8192, 35.76, 75.68, 69.0

    # SMALL_FONT_SIZE = 10
    # MEDIUM_FONT_SIZE = 12

    # fig, axs = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"wspace": 0.35, "hspace": 0.3})
    # # to not cut the labels
    # plt.subplots_adjust(bottom=0.15, top=0.90)
    # sns.set_theme(style="whitegrid", palette="tab10")
    # sns.set_context("poster", rc={
    #     "axes.titlesize": MEDIUM_FONT_SIZE,
    #     "axes.labelsize": MEDIUM_FONT_SIZE,
    #     "xtick.labelsize": SMALL_FONT_SIZE,
    #     "ytick.labelsize": SMALL_FONT_SIZE,
    #     "legend.fontsize": SMALL_FONT_SIZE,
    #     "legend.title_fontsize": SMALL_FONT_SIZE,
    # })

    # # # sns.barplot(
    # # #     data=tdf, x="chunk_size", y="moe_duration",
    # # #     # hue_order=[512, 1024, 2048, 4096, 8192],
    # # #     # palette=["#739BC6", "#FF8E1D", "#2ca02c", "#d62728"],
    # # #     palette=["#FF8E1D", "#979797", "#979797", "#979797", "#739BC6"],
    # # #     ax=axs[0],
    # # #     legend=False,
    # # # )
    # # # axs[0].set_xlabel("Chunk size (tokens)", fontsize=40)
    # # # axs[0].set_ylabel("Duration (ms)", fontsize=40)
    # # # axs[0].set_title("MoE Duration")
    # # # axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=40)
    # # # axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=40)
    # data = [
    #     (512,  764.1, "Total MoE load"),
    #     (512,  764.1/16, "Per-chunk MoE load"),
    #     (1024, 406.3, "Total MoE load"),
    #     (1024, 406.3/8, "Per-chunk MoE load"),
    #     (2048, 213.3, "Total MoE load"),
    #     (2048, 213.3/4, "Per-chunk MoE load"),
    #     (4096, 110.1, "Total MoE load"),
    #     (4096, 110.1/2, "Per-chunk MoE load"),
    #     (8192,  57.1, "Total MoE load"),
    #     (8192,  57.1/1, "Per-chunk MoE load"),
    # ]

    # mdf = pd.DataFrame(data, columns=["chunk_size", "MoE load (GB)", "type"]).sort_values("chunk_size")

    # x = np.arange(len(mdf))


    # def set_ax_spine(ax, is_second_y=False, is_attainment=False):
    #     ax.spines["left"].set_visible(True)
    #     ax.spines["left"].set_linewidth(0.9)
    #     ax.spines["left"].set_color("#000000")
    #     ax.spines["bottom"].set_visible(True)
    #     ax.spines["bottom"].set_linewidth(0.9)
    #     ax.spines["bottom"].set_color("#000000")
    #     if not is_attainment:
    #         ax.spines["right"].set_visible(True)
    #         ax.spines["right"].set_linewidth(0.9)
    #         ax.spines["right"].set_color("#000000")
    #     else:
    #         ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     left = False if not is_attainment else True
    #     right = True if not is_attainment else False
    #     ax.tick_params(axis="y", which="major", width=0.9, length=6, color="#000000", left=left, right=right, bottom=False, top=False)
    #     ax.tick_params(axis="x", which="major", width=0.9, length=3, color="#000000", left=False, right=False, bottom=True, top=False)

    # g = sns.barplot(
    #     data=mdf, x="chunk_size", y="MoE load (GB)",
    #     # kind="bar",
    #     hue="type",
    #     hue_order=["Total MoE load", "Per-chunk MoE load"],
    #     palette=["#FF8E1D", "#979797"],
    #     ax=axs[0],
    #     legend=True,
    #     edgecolor="black", linewidth=0.6,
    # )
    # handles, _labels = axs[0].get_legend_handles_labels()
    # g.legend(handles, _labels, loc='upper right', ncol=1, frameon=False, fontsize=SMALL_FONT_SIZE, title_fontsize=SMALL_FONT_SIZE, title="")

    # # # draw bar (per chunk)
    # # for i in range(len(mdf)):
    # #     axs[0].bar(x[i], mdf.iloc[i]["Per-chunk MoE load"], color="black", alpha=0.3, hatch="xx")

    # # labels
    # # axs[0].set_xticks(x)
    # # axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=SMALL_FONT_SIZE)
    # # axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=SMALL_FONT_SIZE)
    # axs[0].set_xlabel("Chunk size (tokens)", fontsize=MEDIUM_FONT_SIZE)
    # axs[0].set_ylabel("MoE load (GB)", fontsize=MEDIUM_FONT_SIZE)
    # axs[0].set_title("MoE Load", fontsize=MEDIUM_FONT_SIZE)
    # axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # # # value labels on top
    # # axs[0].set_ylim(0, vals.max() * 1.15)
    # # for rect, v in zip(bars, vals):
    # #     axs[0].text(rect.get_x() + rect.get_width()/2, v, f"{v:.1f}",
    # #             ha="center", va="bottom", fontsize=MEDIUM_FONT_SIZE)

    # # grid (y-axis only)
    # axs[0].grid(axis="y", linestyle="-", linewidth=0.5)
    # axs[0].grid(axis="x", visible=False)
    # set_ax_spine(axs[0], is_second_y=False, is_attainment=True)

    # # ax2 = axs[0].twinx()

    # # mdf = mdf.reset_index(drop=True).reset_index()
    # # sns.lineplot(
    # #     data=mdf, x="index", y="Duration",
    # #     color="black",
    # #     ax=ax2, legend=False, marker='o', linestyle='--'
    # # )
    # # ax2.grid(False)
    # # ax2.set_facecolor("none")

    # # sns.barplot(
    # #     data=tdf, x="model", y="attn_duration",
    # #     hue="chunk_size", hue_order=[512, 1024, 2048, 4096],
    # #     # palette=["#739BC6", "#FF8E1D", "#2ca02c", "#d62728"],
    # #     ax=axs[1],
    # # )
    # # axs[1].set_xlabel("")
    # # axs[1].set_ylabel("")
    # # axs[1].set_title("Attention Duration")
    # # axs[1].legend(title="Chunk Size")
    # # axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=40)
    # # axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=40)

    # # 논문용 무난한 컬러 팔레트
    # palette = {
    #     "Attention": "#76B7B2",  # blue
    #     "MoE":       "#E38E39",  # orange
    #     "Etc":       "#979797",  # green
    #     "Non-attn": "#D9D9D9",  # gray
    #     "Communication": "#C5C5C5",  # light gray
    # }

    # # tdf["other_duration"] = tdf["attn_duration"] + tdf["nonattn_duration"] + tdf["etc_duration"]

    # # 해치 패턴 (요청: "//", "\\\\", "xx")
    # hatches = {
    #     "Attention": "////",
    #     "MoE":       r"\\\\",
    #     "Etc":       r"xxxx",
    # }

    # # categories = [("Etc", "etc"), ("MoE", "moe"), ("Attention", "attn")]
    # categories = [("MoE", "moe_duration"), ("Attention", "attn_duration"), ("Non-attn", "nonattn_duration"), ("Communication", "nccl_duration"), ("Etc", "etc_duration")]
    # x_labels = tdf["chunk_size"].astype(str).tolist()
    # x = np.arange(len(tdf))
    # bottom = np.zeros(len(tdf), dtype=float)

    # print(tdf)

    # for label, col in categories:
    #     vals = tdf[col].to_numpy()
    #     axs[1].bar(
    #         x, vals,
    #         bottom=bottom,
    #         label=label,
    #         color=palette[label],
    #         # hatch=hatches[label],
    #         edgecolor="black",
    #         linewidth=0.6,
    #         # alpha=0.9,
    #     )
    #     bottom += vals

    # # --- Labels & layout ---
    # axs[1].set_xticks(x)
    # axs[1].set_xticklabels(x_labels)
    # axs[1].set_xlabel("Chunk size (tokens)", fontsize=MEDIUM_FONT_SIZE)
    # axs[1].set_ylabel("Runtime (ms)", fontsize=MEDIUM_FONT_SIZE)
    # axs[1].set_title("Runtime per Operation", fontsize=MEDIUM_FONT_SIZE)
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=SMALL_FONT_SIZE)
    # axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=SMALL_FONT_SIZE)
    # axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # # y축 그리드만
    # axs[1].grid(axis="y", linestyle="-", linewidth=0.5)
    # axs[1].grid(axis="x", visible=False)
    # set_ax_spine(axs[1], is_second_y=False, is_attainment=True)

    # # 범례
    # axs[1].legend(frameon=False, ncol=1, loc="upper right")

    # plt.savefig("duration_vs_chunk_size.pdf")

    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    sns.set_theme(style="whitegrid")
    SMALL_FONT_SIZE = 10
    MEDIUM_FONT_SIZE = 12
    sns.set_context("poster", rc={
        "axes.titlesize": MEDIUM_FONT_SIZE,
        "axes.labelsize": MEDIUM_FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "legend.title_fontsize": SMALL_FONT_SIZE,
    })

    def set_ax_spine(ax, is_second_y=False, is_attainment=False):
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.spines["bottom"].set_color("#000000")
        if not is_attainment:
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_linewidth(0.9)
            ax.spines["right"].set_color("#000000")
        else:
            ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        left = False if not is_attainment else True
        right = True if not is_attainment else False
        ax.tick_params(axis="y", which="major", width=0.9, length=6, color="#000000", left=left, right=right, bottom=False, top=False)
        ax.tick_params(axis="x", which="major", width=0.9, length=3, color="#000000", left=False, right=False, bottom=True, top=False)

    ax2s = []
    for model_idx, model_name in enumerate(["qwen", "gpt"]):
        short_model_name = "Qwen3-30B-A3B" if model_name == "qwen" else "GPT-OSS-20B"
        arxiv_figure_label = chr(ord('a') + model_idx)
        sharegpt_figure_label = chr(ord('a') + model_idx + 2)
    # for model_idx, model_name in enumerate(["qwen"]):
        layered_prefill_num_stages = 16 if model_name == "qwen" else 12

        # --- Arxiv ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "arxiv")
            & (
                ((df["schedule_mode"] == "layered-prefill") & (df["num_stages"] == layered_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
        ]

        ax = sns.barplot(
            graph_df, x="request_rate", y="slo",
            hue="schedule_mode", hue_order=["chunked-prefill", "layered-prefill"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            legend=False, ax=axs[0, model_idx],
            alpha=0.85, linewidth=1.0, edgecolor="black",
        )
        ax.set_xlabel("", fontsize=MEDIUM_FONT_SIZE)
        if model_idx == 0:
            ax.set_ylabel("SLO Attainment (%)", fontsize=MEDIUM_FONT_SIZE)
        else:
            ax.set_ylabel("")
        ax.set_title(f"$\\bf{{({arxiv_figure_label})}}$ arXiv - {short_model_name}")
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=SMALL_FONT_SIZE, va="top")
        if model_idx == 0:
            ax.set_yticklabels(ax.get_yticks(), fontsize=SMALL_FONT_SIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        else:
            # disable y ticks
            ax.set_yticklabels([""] * len(ax.get_yticks()), fontsize=SMALL_FONT_SIZE)
        # draw line at y = 90
        ax.axhline(90, color='#C23B22', linestyle='--', linewidth=2)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")
        set_ax_spine(ax, is_second_y=False, is_attainment=False)

        ax2 = ax.twinx()
        mean_requests_df = pd.concat([
            graph_df[graph_df["schedule_mode"] == "layered-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])

        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "layered-prefill"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            ax=ax2, legend=True, marker='o', linestyle='--',
            alpha=0.85, linewidth=1.0, markersize=6, markeredgecolor="black", markeredgewidth=0.5

        )
        if model_idx == 1:
            ax2.set_ylabel("Average decode batch size")
        else:
            ax2.set_ylabel("")
        ax2.legend(title="Schedule Mode")
        ax2.grid(False)
        ax2.set_facecolor("none")
        sns.move_legend(ax2, "upper right")
        ax2s.append(ax2)
        set_ax_spine(ax2)

        # --- ShareGPT ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "sharegpt")
            # & (df["max_num_seqs"] == (128 if model_name == "qwen" else 64))
            & (
                ((df["schedule_mode"] == "layered-prefill") & (df["num_stages"] == layered_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 100)
        ]

        ax = sns.barplot(
            graph_df, x="request_rate", y="slo",
            hue="schedule_mode", hue_order=["chunked-prefill", "layered-prefill"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            legend=False, ax=axs[1, model_idx],
            alpha=0.85, linewidth=1.0, edgecolor="black",
        )
        ax.set_xlabel("Request rate (req/s)", fontsize=MEDIUM_FONT_SIZE)
        if model_idx == 0:
            ax.set_ylabel("SLO Attainment (%)", fontsize=MEDIUM_FONT_SIZE)
        else:
            ax.set_ylabel("")
        ax.set_title(f"$\\bf{{({sharegpt_figure_label})}}$ ShareGPT - {short_model_name}")
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=SMALL_FONT_SIZE)
        if model_idx == 0:
            ax.set_yticklabels(ax.get_yticks(), fontsize=SMALL_FONT_SIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        else:
            # disable y ticks
            ax.set_yticklabels([""] * len(ax.get_yticks()), fontsize=SMALL_FONT_SIZE)
        # draw line at y = 90
        ax.axhline(90, color='#C23B22', linestyle='--', linewidth=2)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")
        set_ax_spine(ax, is_second_y=True, is_attainment=False)

        ax2 = ax.twinx()
        mean_requests_df = pd.concat([
            graph_df[graph_df["schedule_mode"] == "layered-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])
        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "layered-prefill"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            ax=ax2, legend=True, marker='o', linestyle='--',
            alpha=0.85, linewidth=1.0, markersize=6, markeredgecolor="black", markeredgewidth=0.5
        )
        if model_idx == 1:
            ax2.set_ylabel("Average decode batch size")
        else:
            ax2.set_ylabel("")
        ax2.legend(title="Schedule Mode")
        ax2.grid(False)
        ax2.set_facecolor("none")
        sns.move_legend(ax2, "upper right")
        ax2s.append(ax2)
        set_ax_spine(ax2)

    handles, labels = ax2s[0].get_legend_handles_labels()
    print(labels)
    _labels = []
    for label in labels:
        if "layered-prefill" == label:
            _labels.append("Layered prefill")
        elif "chunked-prefill" == label:
            _labels.append("Chunked prefill")
    fig.legend(handles, _labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4, frameon=False, fontsize=MEDIUM_FONT_SIZE, title_fontsize=MEDIUM_FONT_SIZE, title="")
    for ax in ax2s:
        ax.legend_.remove()

    # text box below the plots
    # textstr = "(a) Qwen3-30B-A3B                 (b) GPT-OSS-20B "
    # fig.text(0.5, 0.02, textstr, fontsize=60, ha='center', va='center')

    plt.savefig(f"slo_distribution.pdf")

    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    sns.set_theme(style="whitegrid")
    sns.set_context("poster", rc={
        "axes.titlesize": MEDIUM_FONT_SIZE,
        "axes.labelsize": MEDIUM_FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "legend.title_fontsize": SMALL_FONT_SIZE,
    })

    for model_idx, model_name in enumerate(["qwen", "gpt"]):
    # for model_idx, model_name in enumerate(["qwen"]):
        short_model_name = "Qwen3-30B-A3B" if model_name == "qwen" else "GPT-OSS-20B"
        arxiv_figure_label = chr(ord('a') + model_idx)
        sharegpt_figure_label = chr(ord('a') + model_idx + 2)
        layered_prefill_num_stages = 16 if model_name == "qwen" else 12
        # --- Arxiv ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "arxiv")
            & (
                ((df["schedule_mode"] == "layered-prefill") & (df["num_stages"] == layered_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
        ]

        data = []
        for row in graph_df.itertuples():
            data.append({
                "request_rate": row.request_rate,
                "value": row.ttft_slo_attain,
                "type": "Layered prefill TTFT" if row.schedule_mode == "layered-prefill" else "Chunked prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered prefill TBT" if row.schedule_mode == "layered-prefill" else "Chunked prefill TBT",
                # "schedule_mode": row.schedule_mode,
            })
        temp_df = pd.DataFrame(data)

        g = sns.lineplot(
            data=temp_df, x="request_rate", y="value",
            hue="type", hue_order=["Chunked prefill TTFT", "Layered prefill TTFT", "Chunked prefill TBT", "Layered prefill TBT"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            style="type", markers=True,
            dashes=[(2, 0), (2, 0), (2, 2), (2, 2)],
            style_order=["Chunked prefill TTFT", "Layered prefill TTFT", "Chunked prefill TBT", "Layered prefill TBT"],
            ax=axs[0, model_idx],
            alpha=0.85, linewidth=1.0, markersize=6, markeredgecolor="black", markeredgewidth=0.5,
        )
        g.set_title(f"$\\bf{{({arxiv_figure_label})}}$ arXiv - {short_model_name}")
        g.xaxis.set_major_locator(plt.MultipleLocator((temp_df["request_rate"].max() - temp_df["request_rate"].min()) / 5))
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=SMALL_FONT_SIZE)
        # g.set_yticklabels(g.get_yticks(), fontsize=SMALL_FONT_SIZE)
        g.set_xlabel("", fontsize=MEDIUM_FONT_SIZE)
        if model_idx == 0:
            g.set_ylabel("SLO Attainment (%)", fontsize=MEDIUM_FONT_SIZE)
        else:
            g.set_ylabel("")
        g.set_ylim(0, 105)
        g.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        g.legend(title="Schedule Mode")
        sns.move_legend(g, "upper right")
        set_ax_spine(g, is_second_y=False, is_attainment=True)
        g.yaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")
        g.xaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")

        # --- ShareGPT ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "sharegpt")
            # & (df["max_num_seqs"] == (64 if model_name == "qwen" else 64))
            & (
                ((df["schedule_mode"] == "layered-prefill") & (df["num_stages"] == layered_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 100)
        ]

        data = []
        for row in graph_df.itertuples():
            data.append({
                "request_rate": row.request_rate,
                "value": row.ttft_slo_attain,
                "type": "Layered prefill TTFT" if row.schedule_mode == "layered-prefill" else "Chunked prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered prefill TBT" if row.schedule_mode == "layered-prefill" else "Chunked prefill TBT",
                # "schedule_mode": row.schedule_mode,
            })
        temp_df = pd.DataFrame(data)

        g = sns.lineplot(
            data=temp_df, x="request_rate", y="value",
            hue="type", hue_order=["Chunked prefill TTFT", "Layered prefill TTFT", "Chunked prefill TBT", "Layered prefill TBT"],
            palette=["#FF8E1D", "#739BC6", "#FF8E1D", "#739BC6"],
            style="type", markers=True,
            dashes=[(2, 0), (2, 0), (2, 2), (2, 2)],
            style_order=["Chunked prefill TTFT", "Layered prefill TTFT", "Chunked prefill TBT", "Layered prefill TBT"],
            ax=axs[1, model_idx],
            alpha=0.85, linewidth=1.0, markersize=6, markeredgecolor="black", markeredgewidth=0.5,
        )
        g.set_title(f"$\\bf{{({sharegpt_figure_label})}}$ shareGPT - {short_model_name}")
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=SMALL_FONT_SIZE)
        g.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        # g.set_yticklabels(g.get_yticks(), fontsize=SMALL_FONT_SIZE)
        g.set_xlabel("Request rate (req/s)", fontsize=MEDIUM_FONT_SIZE)
        if model_idx == 0:
            g.set_ylabel("SLO Attainment (%)", fontsize=MEDIUM_FONT_SIZE)
        else:
            g.set_ylabel("")
        g.set_ylim(0, 105)
        g.legend(title="Schedule Mode")
        sns.move_legend(g, "upper right")
        set_ax_spine(g, is_second_y=False, is_attainment=True)
        g.yaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")
        g.xaxis.grid(True, linestyle=":", linewidth=0.9, color="#D0D0D0")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    for ax in axs.ravel():
        if ax.legend_ is not None:
            ax.legend_.remove()

    # textstr = "(a) Qwen3-30B-A3B                 (b) GPT-OSS-20B "
    # fig.text(0.5, 0.02, textstr, fontsize=60, ha='center', va='center')

    plt.savefig(f"slo_attainment.pdf")

    layered_prefill_json_file = "logs/benchmark_qwen_8192_256_32768_0.85_2_False_debug_layered-prefill_16_arxiv_-1_None_1.3_780.json"
    chunked_prefill_json_file = "logs/benchmark_qwen_512_256_32768_0.85_2_False_debug_chunked-prefill_1_arxiv_-1_None_1.3_780.json"

    idx = 205
    with open(layered_prefill_json_file, 'r') as f:
        layered_prefill_data = json.load(f)
        layered_prefill_ttft = layered_prefill_data["ttfts"][idx]
        layered_prefill_itl = layered_prefill_data["itls"][idx]

        layered_prefill_times = np.array([layered_prefill_ttft] + layered_prefill_itl).cumsum()
        layered_prefill_num_tokens = np.arange(1, len(layered_prefill_times) + 1)

    with open(chunked_prefill_json_file, 'r') as f:
        chunked_prefill_data = json.load(f)
        chunked_prefill_ttft = chunked_prefill_data["ttfts"][idx]
        chunked_prefill_itl = chunked_prefill_data["itls"][idx]

        chunked_prefill_times = np.array([chunked_prefill_ttft] + chunked_prefill_itl).cumsum()
        chunked_prefill_num_tokens = np.arange(1, len(chunked_prefill_times) + 1)

    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(bottom=0.15, top=0.90, left=0.15, right=0.95)
    axs.plot(chunked_prefill_times, chunked_prefill_num_tokens, label='Chunked prefill', color="#E38E39")
    axs.plot(layered_prefill_times, layered_prefill_num_tokens, label='Layered prefill', color="#7D9BBC")

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('# of Generated Tokens')
    axs.set_title('Token Generation Over Time')
    axs.grid(axis="y", linestyle=":", linewidth=0.9, color="#D0D0D0")
    axs.grid(axis="x", linestyle=":", linewidth=0.9, color="#D0D0D0")
    axs.set_xlim(left=0)
    axs.set_ylim(bottom=0)
    axs.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    axs.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
    axs.set_xticklabels(axs.get_xticklabels(), rotation=0, fontsize=SMALL_FONT_SIZE)
    axs.set_yticklabels(axs.get_yticks(), fontsize=SMALL_FONT_SIZE)
    set_ax_spine(axs, is_second_y=False, is_attainment=True)
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=1, bbox_to_anchor=(0.95, 0.15), frameon=False, fontsize=MEDIUM_FONT_SIZE, title_fontsize=MEDIUM_FONT_SIZE, title="")
    plt.savefig("token_generation.pdf")
