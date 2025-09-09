import json
import os

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
            FROM cupti_kernel AS cupti_kernel
            WHERE 1=1
                AND cupti_kernel.start_time >= {start}
                AND cupti_kernel.end_time <= {end}
        )
        SELECT *
        FROM kernel
        ORDER BY kernel.start_time
    """)
    return kernel_df


if __name__ == "__main__":
    df = pd.read_csv("benchmark_results.csv")

    print("Model: Qwen3-30B-A3B")
    data = []
    for model in ["Qwen3-30B-A3B", "gpt-oss-20b"]:
        for chunk_size in [512, 1024, 2048, 4096]:
            cur = connect(f"logs/{model}-16384-{chunk_size}.sqlite")
            kernel_df = get_kernel_btw(cur, 0, 1000000000000)
            filtered_kernel_df = kernel_df[kernel_df["start_time"] >= 50e9]
            dist_df = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("dist")]
            start_time = (dist_df.iloc[len(dist_df) // 4 * 3 - 1]["start_time"] + dist_df.iloc[len(dist_df) // 4 * 3]["start_time"]) / 2
            filtered_kernel_df = filtered_kernel_df[filtered_kernel_df["start_time"] >= start_time].sort_values(by="start_time", ascending=True)
            duration = (filtered_kernel_df.iloc[-1]["end_time"] - filtered_kernel_df.iloc[0]["start_time"]) / 1e6
            attn_duration = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("flash")]["duration"].sum() / 1e6  # Convert to ms
            moe_duration = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("moe")]["duration"].sum() / 1e6  # Convert to ms
            nonattn_duration = filtered_kernel_df[~filtered_kernel_df["cupti_kernel_name"].str.contains("flash") & ~filtered_kernel_df["cupti_kernel_name"].str.contains("moe")]["duration"].sum() / 1e6
            etc_duration = duration - attn_duration - moe_duration - nonattn_duration
            # print(nonattn_kernel_df.groupby("cupti_kernel_name")["duration"].sum().sort_values(ascending=True).head(10))
            print(f"Chunk size: {chunk_size}, Attn duration: {attn_duration:.2f} ms, MoE duration: {moe_duration:.2f} ms, Non-attn duration: {nonattn_duration:.2f} ms, Etc duration: {etc_duration:.2f} ms")
            total_chunk_num = 16384 // chunk_size
            for input_length in [4096, 8192, 16384]:
                if input_length < chunk_size:
                    continue
                attn_kernel_df = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("flash")]
                attn_kernel_df = attn_kernel_df.sort_values(by="start_time", ascending=True)
                chunk_num = input_length // chunk_size
                num_attn_call = 48 if model == "Qwen3-30B-A3B" else 24
                assert len(attn_kernel_df) == total_chunk_num * num_attn_call, f"Chunk size {chunk_size} and input length {input_length} mismatch: {len(attn_kernel_df)} != {total_chunk_num * num_attn_call}"
                attn_kernel_df = attn_kernel_df.iloc[:chunk_num * num_attn_call]
                _start_time = attn_kernel_df.iloc[0]["start_time"]
                _end_time = attn_kernel_df.iloc[-1]["end_time"]
                duration = (_end_time - _start_time) / 1e6  # Convert to ms
                moe_kernel_df = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("moe") & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
                nonattn_kernel_df = filtered_kernel_df[~filtered_kernel_df["cupti_kernel_name"].str.contains("flash") & ~filtered_kernel_df["cupti_kernel_name"].str.contains("moe") & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
                attn_duration = attn_kernel_df["duration"].sum() / 1e6  # Convert to ms
                moe_duration = moe_kernel_df["duration"].sum() / 1e6  # Convert to ms
                nonattn_duration = nonattn_kernel_df["duration"].sum() / 1e6
                etc_duration = duration - attn_duration - moe_duration - nonattn_duration

                data.append({
                    "model": "Qwen3-30B-A3B" if model == "Qwen3-30B-A3B" else "GPT-OSS-20B",
                    "chunk_size": chunk_size,
                    "input_length": input_length,
                    "attn_duration": attn_duration,
                    "moe_duration": moe_duration,
                    "nonattn_duration": nonattn_duration,
                    "etc_duration": etc_duration,
                    "total_duration": duration,
                })

    tdf = pd.DataFrame(data)
    tdf = tdf[(tdf["chunk_size"] <= 4096) & tdf["input_length"].isin([16384])]

    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    sns.set_theme(style="whitegrid", palette="tab10")
    sns.set_context("poster", rc={
        "axes.titlesize": 40,
        "axes.labelsize": 40,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
        "legend.title_fontsize": 32,
    })

    sns.barplot(
        data=tdf, x="model", y="attn_duration",
        hue="chunk_size", hue_order=[512, 1024, 2048, 4096],
        # palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        ax=axs[0],
        legend=False,
    )
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Duration (ms)", fontsize=32)
    axs[0].set_title("Attention Duration")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=32)
    axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=32)
    # axs[0].legend(title="Chunk Size")

    sns.barplot(
        data=tdf, x="model", y="moe_duration",
        hue="chunk_size", hue_order=[512, 1024, 2048, 4096],
        # palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        ax=axs[1],
    )
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")
    axs[1].set_title("MoE Duration")
    axs[1].legend(title="Chunk Size")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=32)
    axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=32)

    plt.savefig("duration_vs_chunk_size.pdf")

    # import pdb; pdb.set_trace()

    # model_name = "models--Qwen--Qwen3-30B-A3B"
    # # model_name = "models--Qwen--Qwen3-8B"

    # benchmark_dataset, request_rate, max_num_batched_tokens = ("longbench", 0.15, 256)
    # # benchmark_dataset, request_rate, max_num_batched_tokens = ("sharegpt", 1.0, 1024)

    # chunked_prefill_p99_ttfts = []
    # chunked_prefill_p99_itls = []
    # chunked_prefill_slos = []

    # for chunk_size in [512, 1024, 2048, 4096]:
    #     filter = (df["dataset_name"] == benchmark_dataset) & (df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == chunk_size) & (df["request_rate"] == request_rate) & (df["model_name"] == model_name)
    #     chunked_prefill_ttfts = df[filter]["p99_ttft"].values.tolist()
    #     chunked_prefill_itls = df[filter]["p99_itl"].values.tolist()
    #     chunked_prefill_slo = df[filter]["slo"].values.tolist()
    #     if len(chunked_prefill_ttfts) == 1:
    #         chunked_prefill_p99_ttfts.append(chunked_prefill_ttfts[0])
    #         chunked_prefill_p99_itls.append(chunked_prefill_itls[0])
    #         chunked_prefill_slos.append(chunked_prefill_slo[0])
    #     else:
    #         print(f"chunk_size {chunk_size} has no or multiple entries: {chunked_prefill_ttfts}")

    # print(chunked_prefill_p99_ttfts)
    # print(chunked_prefill_p99_itls)
    # print(chunked_prefill_slos)

    # staged_prefill_p99_ttfts = []
    # staged_prefill_p99_itls = []
    # staged_prefill_slos = []

    # for num_stage in [1, 2, 4, 8, 16]:
    #     filter = (df["dataset_name"] == benchmark_dataset) & (df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == num_stage) & (df["request_rate"] == request_rate) & (df["model_name"] == model_name) & (df["max_num_batched_tokens"] == max_num_batched_tokens)
    #     staged_prefill_ttfts = df[filter]["p99_ttft"].values.tolist()
    #     staged_prefill_itls = df[filter]["p99_itl"].values.tolist()
    #     staged_prefill_slo = df[filter]["slo"].values.tolist()
    #     if len(staged_prefill_ttfts) == 1:
    #         staged_prefill_p99_ttfts.append(staged_prefill_ttfts[0])
    #         staged_prefill_p99_itls.append(staged_prefill_itls[0])
    #         staged_prefill_slos.append(staged_prefill_slo[0])
    #     else:
    #         print(f"num_stage {num_stage} has no or multiple entries: {staged_prefill_ttfts}")

    # print(staged_prefill_p99_ttfts)
    # print(staged_prefill_p99_itls)
    # print(staged_prefill_slos)

    # plt.figure(figsize=(12, 6))
    # plt.plot(chunked_prefill_p99_itls, chunked_prefill_p99_ttfts, marker='o', label='Chunked Prefill')
    # plt.plot(staged_prefill_p99_itls, staged_prefill_p99_ttfts, marker='o', label='Staged Prefill')

    # plt.xlabel('P99 ITL (s)')
    # plt.ylabel('P99 TTFT (s)')
    # plt.title(f'P99 TTFT vs P99 ITL for {model_name} on {benchmark_dataset} at request rate {request_rate}')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"p99_ttft_vs_itl_{model_name.replace('--', '_')}_{benchmark_dataset}_request_rate_{request_rate}.pdf")

    # 16384 1394.94 8192 1479.21 4096 1563.45 2048 1782.23 1024 2625.53 512 4413.58

    chunk_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    latency = [4413.58, 2625.53, 1782.23, 1563.45, 1479.21, 1394.94]
    latency = [3404.28, 1974.99, 1757.37, 1652.51, 1607.98, 1591.23]
    num_chunks = [16384 // chunk_size for chunk_size in chunk_sizes]

    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(24, 24))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    sns.set_theme(style="whitegrid")
    sns.set_context("poster", rc={
        "axes.titlesize": 40,
        "axes.labelsize": 40,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
        "legend.title_fontsize": 32,
    })

    ax2s = []
    for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B", "models--openai--gpt-oss-20b"]):
    # for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B"]):
        staged_prefill_num_stages = 16 if model_name == "models--Qwen--Qwen3-30B-A3B" else 12

        # --- LongBench ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "arxiv")
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
        ]

        ax = sns.barplot(
            graph_df, x="request_rate", y="slo",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
            legend=False, ax=axs[0, model_idx]
        )
        ax.set_xlabel("Request Rate", fontsize=40)
        if model_idx == 0:
            ax.set_ylabel("SLO Attainment (%)", fontsize=40)
        else:
            ax.set_ylabel("")
        ax.set_title(f"Arxiv")
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=32)
        ax.set_yticklabels(ax.get_yticks(), fontsize=32)

        ax2 = ax.twinx()
        mean_requests_df = pd.concat([
            graph_df[graph_df["schedule_mode"] == "staged-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])

        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
            ax=ax2, legend=True, marker='o', linestyle='--'
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

        # --- ShareGPT ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "sharegpt")
            & (df["max_num_seqs"] == (64 if model_name == "models--Qwen--Qwen3-30B-A3B" else 64))
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 600)
        ]

        ax = sns.barplot(
            graph_df, x="request_rate", y="slo",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
            legend=False, ax=axs[1, model_idx]
        )
        ax.set_xlabel("Request Rate", fontsize=40)
        if model_idx == 0:
            ax.set_ylabel("SLO Attainment (%)", fontsize=40)
        else:
            ax.set_ylabel("")
        ax.set_title(f"ShareGPT")
        ax.set_ylim(0, 100)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=32)
        ax.set_yticklabels(ax.get_yticks(), fontsize=32)

        ax2 = ax.twinx()
        mean_requests_df = pd.concat([
            graph_df[graph_df["schedule_mode"] == "staged-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])
        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
            ax=ax2, legend=True, marker='o', linestyle='--'
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

    handles, labels = ax2s[0].get_legend_handles_labels()
    print(labels)
    _labels = []
    for label in labels:
        if "staged-prefill" == label:
            _labels.append("Layered Prefill")
        elif "chunked-prefill" == label:
            _labels.append("Chunked Prefill")
    fig.legend(handles, _labels, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4, frameon=False, fontsize=40, title_fontsize=48, title="Schedule Mode")
    for ax in ax2s:
        ax.legend_.remove()

    # text box below the plots
    textstr = "(a) Qwen3-30B-A3B                 (b) GPT-OSS-20B "
    fig.text(0.5, 0.02, textstr, fontsize=60, ha='center', va='center')

    plt.savefig(f"slo_distribution.pdf")

    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2, 2, figsize=(24, 24))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    sns.set_theme(style="whitegrid")
    sns.set_context("poster", rc={
        "axes.titlesize": 40,
        "axes.labelsize": 40,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
        "legend.title_fontsize": 32,
    })

    for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B", "models--openai--gpt-oss-20b"]):
    # for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B"]):
        staged_prefill_num_stages = 16 if model_name == "models--Qwen--Qwen3-30B-A3B" else 12
        # --- LongBench ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "arxiv")
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
        ]

        data = []
        for row in graph_df.itertuples():
            data.append({
                "request_rate": row.request_rate,
                "value": row.ttft_slo_attain,
                "type": "Layered Prefill TTFT" if row.schedule_mode == "staged-prefill" else "Chunked Prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered Prefill TBT" if row.schedule_mode == "staged-prefill" else "Chunked Prefill TBT",
                # "schedule_mode": row.schedule_mode,
            })
        temp_df = pd.DataFrame(data)

        g = sns.lineplot(
            data=temp_df, x="request_rate", y="value",
            hue="type", hue_order=["Chunked Prefill TTFT", "Layered Prefill TTFT", "Chunked Prefill TBT", "Layered Prefill TBT"],
            palette=["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"],
            style="type", markers=True,
            dashes=[(2, 0), (2, 0), (2, 2), (2, 2)],
            style_order=["Chunked Prefill TTFT", "Layered Prefill TTFT", "Chunked Prefill TBT", "Layered Prefill TBT"],
            ax=axs[0, model_idx]
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=32)
        # g.set_yticklabels(g.get_yticks(), fontsize=32)
        g.set_xlabel("Request Rate", fontsize=40)
        if model_idx == 0:
            g.set_ylabel("SLO Attainment (%)", fontsize=40)
        else:
            g.set_ylabel("")
        g.set_title(f"Arxiv")
        g.set_ylim(0, 100)
        g.legend(title="Schedule Mode")
        sns.move_legend(g, "upper right")

        # --- ShareGPT ---
        graph_df = df[
            (df["model_name"] == model_name)
            & (df["dataset_name"] == "sharegpt")
            & (df["max_num_seqs"] == (64 if model_name == "models--Qwen--Qwen3-30B-A3B" else 64))
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 600)
        ]

        data = []
        for row in graph_df.itertuples():
            data.append({
                "request_rate": row.request_rate,
                "value": row.ttft_slo_attain,
                "type": "Layered Prefill TTFT" if row.schedule_mode == "staged-prefill" else "Chunked Prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered Prefill TBT" if row.schedule_mode == "staged-prefill" else "Chunked Prefill TBT",
                # "schedule_mode": row.schedule_mode,
            })
        temp_df = pd.DataFrame(data)

        g = sns.lineplot(
            data=temp_df, x="request_rate", y="value",
            hue="type", hue_order=["Chunked Prefill TTFT", "Layered Prefill TTFT", "Chunked Prefill TBT", "Layered Prefill TBT"],
            palette=["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"],
            style="type", markers=True,
            dashes=[(2, 0), (2, 0), (2, 2), (2, 2)],
            style_order=["Chunked Prefill TTFT", "Layered Prefill TTFT", "Chunked Prefill TBT", "Layered Prefill TBT"],
            ax=axs[1, model_idx]
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=32)
        # g.set_yticklabels(g.get_yticks(), fontsize=32)
        g.set_xlabel("Request Rate", fontsize=40)
        if model_idx == 0:
            g.set_ylabel("SLO Attainment (%)", fontsize=40)
        else:
            g.set_ylabel("")
        g.set_title(f"ShareGPT")
        g.set_ylim(0, 100)
        g.legend(title="Schedule Mode")
        sns.move_legend(g, "upper right")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)
    for ax in axs.ravel():
        if ax.legend_ is not None:
            ax.legend_.remove()

    textstr = "(a) Qwen3-30B-A3B                 (b) GPT-OSS-20B "
    fig.text(0.5, 0.02, textstr, fontsize=60, ha='center', va='center')

    plt.savefig(f"slo_attainment.pdf")

    staged_prefill_json_file = "logs/benchmark_models--Qwen--Qwen3-30B-A3B_8192_64_32768_0.9_1_False_debug_staged-prefill_16_sharegpt_-1_None_1_600.json"
    chunked_prefill_json_file = "logs/benchmark_models--Qwen--Qwen3-30B-A3B_512_64_32768_0.9_1_False_debug_chunked-prefill_1_sharegpt_-1_None_1_600.json"

    idx = 305
    with open(staged_prefill_json_file, 'r') as f:
        staged_prefill_data = json.load(f)
        staged_prefill_ttft = staged_prefill_data["ttfts"][idx]
        staged_prefill_itl = staged_prefill_data["itls"][idx]

        staged_prefill_times = np.array([staged_prefill_ttft] + staged_prefill_itl).cumsum()
        staged_prefill_num_tokens = np.arange(1, len(staged_prefill_times) + 1)

    with open(chunked_prefill_json_file, 'r') as f:
        chunked_prefill_data = json.load(f)
        chunked_prefill_ttft = chunked_prefill_data["ttfts"][idx]
        chunked_prefill_itl = chunked_prefill_data["itls"][idx]

        chunked_prefill_times = np.array([chunked_prefill_ttft] + chunked_prefill_itl).cumsum()
        chunked_prefill_num_tokens = np.arange(1, len(chunked_prefill_times) + 1)

    plt.clf()
    plt.cla()
    plt.figure(figsize=(18, 12))
    plt.plot(chunked_prefill_times, chunked_prefill_num_tokens, marker='o', label='Chunked Prefill', color="#1f77b4")
    plt.plot(staged_prefill_times, staged_prefill_num_tokens, marker='o', label='Layered Prefill', color="#ff7f0e")

    plt.xlabel('Time (s)')
    plt.ylabel('# of Generated Tokens')
    plt.title('Token Generation Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("token_generation.pdf")
