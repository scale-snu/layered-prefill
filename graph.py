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
                AND cupti_kernel.deviceId = 0
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
    for model in ["Qwen3-30B-A3B"]:
        for chunk_size in [512, 1024, 2048, 4096, 8192]:
            print(chunk_size)
            cur = connect(f"logs/{model}-{chunk_size}.sqlite")
            kernel_df = get_kernel_btw(cur, 0, 1000000000000)
            filtered_kernel_df = kernel_df[kernel_df["start_time"] >= (50e9 if model == "Qwen3-30B-A3B" else 40e9)]
            flash_df = filtered_kernel_df[filtered_kernel_df["cupti_kernel_name"].str.contains("flash")]
            start_time = (flash_df.iloc[len(flash_df) // 3 * 2 - 1]["start_time"] + flash_df.iloc[len(flash_df) // 3 * 2]["start_time"]) / 2
            # import pdb; pdb.set_trace()
            filtered_kernel_df = filtered_kernel_df[filtered_kernel_df["start_time"] >= start_time].sort_values(by="start_time", ascending=True)
            duration = (filtered_kernel_df.iloc[-1]["end_time"] - filtered_kernel_df.iloc[0]["start_time"]) / 1e6
            nccl_filter = filtered_kernel_df["cupti_kernel_name"].str.contains("nccl")
            nccl_duration = filtered_kernel_df[nccl_filter]["duration"].sum() / 1e6  # Convert to ms
            attn_filter = (
                filtered_kernel_df["cupti_kernel_name"].str.contains("flash")
                & ~filtered_kernel_df["cupti_kernel_name"].str.contains("prepare")
                & ~filtered_kernel_df["cupti_kernel_name"].str.contains("Combine")
            )
            attn_duration = filtered_kernel_df[attn_filter]["duration"].sum() / 1e6  # Convert to ms
            moe_filter = filtered_kernel_df["cupti_kernel_name"].str.contains("fused_moe_kernel")
            moe_duration = filtered_kernel_df[moe_filter]["duration"].sum() / 1e6  # Convert to ms
            nonattn_duration = filtered_kernel_df[~nccl_filter & ~attn_filter & ~moe_filter]["duration"].sum() / 1e6
            etc_duration = duration - nccl_duration - attn_duration - moe_duration - nonattn_duration
            # print(nonattn_kernel_df.groupby("cupti_kernel_name")["duration"].sum().sort_values(ascending=True).head(10))
            print(f"Chunk size: {chunk_size}, Nccl duration: {nccl_duration:.2f}, Attn duration: {attn_duration:.2f} ms, MoE duration: {moe_duration:.2f} ms, Non-attn duration: {nonattn_duration:.2f} ms, Etc duration: {etc_duration:.2f} ms")
            total_chunk_num = 8192 // chunk_size
            for input_length in [4096, 8192]:
                if input_length < chunk_size:
                    continue
                attn_kernel_df = filtered_kernel_df[attn_filter]
                attn_kernel_df = attn_kernel_df.sort_values(by="start_time", ascending=True)
                chunk_num = input_length // chunk_size
                num_attn_call = 48 if model == "Qwen3-30B-A3B" else 24
                assert len(attn_kernel_df) == total_chunk_num * num_attn_call, f"Chunk size {chunk_size} and input length {input_length} mismatch: {len(attn_kernel_df)} != {total_chunk_num * num_attn_call}"
                attn_kernel_df = attn_kernel_df.iloc[:chunk_num * num_attn_call]
                _start_time = attn_kernel_df.iloc[0]["start_time"]
                _end_time = attn_kernel_df.iloc[-1]["end_time"]
                duration = (_end_time - _start_time) / 1e6  # Convert to ms
                moe_kernel_df = filtered_kernel_df[moe_filter & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
                nonattn_kernel_df = filtered_kernel_df[~attn_filter & ~moe_filter & (filtered_kernel_df["start_time"] >= _start_time) & (filtered_kernel_df["end_time"] <= _end_time)]
                attn_duration = attn_kernel_df["duration"].sum() / 1e6  # Convert to ms
                moe_duration = moe_kernel_df["duration"].sum() / 1e6  # Convert to ms
                nonattn_duration = nonattn_kernel_df["duration"].sum() / 1e6
                etc_duration = duration - attn_duration - moe_duration - nonattn_duration

                data.append({
                    "model": model,
                    "chunk_size": chunk_size,
                    "input_length": input_length,
                    "attn_duration": attn_duration,
                    "moe_duration": moe_duration,
                    "nonattn_duration": nonattn_duration,
                    "etc_duration": etc_duration,
                    "total_duration": duration,
                })

    tdf = pd.DataFrame(data)
    tdf = tdf[(tdf["chunk_size"] <= 8192) & tdf["input_length"].isin([8192])]

    # # 8192: 57.1, 4096: 110.1, 2048: 213.3, 1024: 406.3, 512: 764.1
    # # chunk size, the number of chunks, MoE load Bytes (GB), MoE duration (ms)
    # # 512, 16, 764.1, 311.95
    # # 1024, 8, 406.3, 184.88
    # # 2048, 4, 213.3, 112.28
    # # 4096, 2, 110.1, 84.57
    # # 8192, 1, 57.1, 75.68
    # # chunk size, attn duration, moe duration, etc duration
    # # 512, 68.975, 311.95, 104.645
    # # 8192, 35.76, 75.68, 69.0

    SMALL_FONT_SIZE = 10
    MEDIUM_FONT_SIZE = 12

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"wspace": 0.35, "hspace": 0.3})
    # to not cut the labels
    plt.subplots_adjust(bottom=0.15, top=0.90)
    sns.set_theme(style="whitegrid", palette="tab10")
    sns.set_context("poster", rc={
        "axes.titlesize": MEDIUM_FONT_SIZE,
        "axes.labelsize": MEDIUM_FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "legend.title_fontsize": SMALL_FONT_SIZE,
    })

    # # sns.barplot(
    # #     data=tdf, x="chunk_size", y="moe_duration",
    # #     # hue_order=[512, 1024, 2048, 4096, 8192],
    # #     # palette=["#739BC6", "#FF8E1D", "#2ca02c", "#d62728"],
    # #     palette=["#FF8E1D", "#979797", "#979797", "#979797", "#739BC6"],
    # #     ax=axs[0],
    # #     legend=False,
    # # )
    # # axs[0].set_xlabel("Chunk size (tokens)", fontsize=40)
    # # axs[0].set_ylabel("Duration (ms)", fontsize=40)
    # # axs[0].set_title("MoE Duration")
    # # axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=40)
    # # axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=40)
    data = [
        (512,  764.1, "Total MoE load"),
        (512,  764.1/16, "Per-chunk MoE load"),
        (1024, 406.3, "Total MoE load"),
        (1024, 406.3/8, "Per-chunk MoE load"),
        (2048, 213.3, "Total MoE load"),
        (2048, 213.3/4, "Per-chunk MoE load"),
        (4096, 110.1, "Total MoE load"),
        (4096, 110.1/2, "Per-chunk MoE load"),
        (8192,  57.1, "Total MoE load"),
        (8192,  57.1/1, "Per-chunk MoE load"),
    ]

    mdf = pd.DataFrame(data, columns=["chunk_size", "MoE load (GB)", "type"]).sort_values("chunk_size")

    x = np.arange(len(mdf))


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

    g = sns.barplot(
        data=mdf, x="chunk_size", y="MoE load (GB)",
        # kind="bar",
        hue="type",
        hue_order=["Total MoE load", "Per-chunk MoE load"],
        palette=["#FF8E1D", "#979797"],
        ax=axs[0],
        legend=True,
        edgecolor="black", linewidth=0.6,
    )
    handles, _labels = axs[0].get_legend_handles_labels()
    g.legend(handles, _labels, loc='upper right', ncol=1, frameon=False, fontsize=SMALL_FONT_SIZE, title_fontsize=SMALL_FONT_SIZE, title="")

    # # draw bar (per chunk)
    # for i in range(len(mdf)):
    #     axs[0].bar(x[i], mdf.iloc[i]["Per-chunk MoE load"], color="black", alpha=0.3, hatch="xx")

    # labels
    # axs[0].set_xticks(x)
    # axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=SMALL_FONT_SIZE)
    # axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=SMALL_FONT_SIZE)
    axs[0].set_xlabel("Chunk size (tokens)", fontsize=MEDIUM_FONT_SIZE)
    axs[0].set_ylabel("MoE load (GB)", fontsize=MEDIUM_FONT_SIZE)
    axs[0].set_title("MoE Load", fontsize=MEDIUM_FONT_SIZE)
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # # value labels on top
    # axs[0].set_ylim(0, vals.max() * 1.15)
    # for rect, v in zip(bars, vals):
    #     axs[0].text(rect.get_x() + rect.get_width()/2, v, f"{v:.1f}",
    #             ha="center", va="bottom", fontsize=MEDIUM_FONT_SIZE)

    # grid (y-axis only)
    axs[0].grid(axis="y", linestyle="-", linewidth=0.5)
    axs[0].grid(axis="x", visible=False)
    set_ax_spine(axs[0], is_second_y=False, is_attainment=True)

    # ax2 = axs[0].twinx()

    # mdf = mdf.reset_index(drop=True).reset_index()
    # sns.lineplot(
    #     data=mdf, x="index", y="Duration",
    #     color="black",
    #     ax=ax2, legend=False, marker='o', linestyle='--'
    # )
    # ax2.grid(False)
    # ax2.set_facecolor("none")

    # sns.barplot(
    #     data=tdf, x="model", y="attn_duration",
    #     hue="chunk_size", hue_order=[512, 1024, 2048, 4096],
    #     # palette=["#739BC6", "#FF8E1D", "#2ca02c", "#d62728"],
    #     ax=axs[1],
    # )
    # axs[1].set_xlabel("")
    # axs[1].set_ylabel("")
    # axs[1].set_title("Attention Duration")
    # axs[1].legend(title="Chunk Size")
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=40)
    # axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=40)

    # 논문용 무난한 컬러 팔레트
    palette = {
        "Attention": "#76B7B2",  # blue
        "MoE":       "#E38E39",  # orange
        "Etc":       "#979797",  # green
    }

    tdf["other_duration"] = tdf["attn_duration"] + tdf["nonattn_duration"] + tdf["etc_duration"]

    # 해치 패턴 (요청: "//", "\\\\", "xx")
    hatches = {
        "Attention": "////",
        "MoE":       r"\\\\",
        "Etc":       r"xxxx",
    }

    # categories = [("Etc", "etc"), ("MoE", "moe"), ("Attention", "attn")]
    categories = [("Etc", "other_duration"), ("MoE", "moe_duration")]
    x_labels = tdf["chunk_size"].astype(str).tolist()
    x = np.arange(len(tdf))
    bottom = np.zeros(len(tdf), dtype=float)

    for label, col in categories:
        vals = tdf[col].to_numpy()
        axs[1].bar(
            x, vals,
            bottom=bottom,
            label=label,
            color=palette[label],
            # hatch=hatches[label],
            edgecolor="black",
            linewidth=0.6,
            # alpha=0.9,
        )
        bottom += vals

    # --- Labels & layout ---
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_xlabel("Chunk size (tokens)", fontsize=MEDIUM_FONT_SIZE)
    axs[1].set_ylabel("Runtime (ms)", fontsize=MEDIUM_FONT_SIZE)
    axs[1].set_title("Runtime per Operation", fontsize=MEDIUM_FONT_SIZE)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=SMALL_FONT_SIZE)
    axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=SMALL_FONT_SIZE)
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

    # y축 그리드만
    axs[1].grid(axis="y", linestyle="-", linewidth=0.5)
    axs[1].grid(axis="x", visible=False)
    set_ax_spine(axs[1], is_second_y=False, is_attainment=True)

    # 범례
    axs[1].legend(frameon=False, ncol=1, loc="upper right")

    plt.savefig("duration_vs_chunk_size.pdf")

    # # model_name = "models--Qwen--Qwen3-30B-A3B"
    # # # model_name = "models--Qwen--Qwen3-8B"

    # # benchmark_dataset, request_rate, max_num_batched_tokens = ("longbench", 0.15, 256)
    # # # benchmark_dataset, request_rate, max_num_batched_tokens = ("sharegpt", 1.0, 1024)

    # # chunked_prefill_p99_ttfts = []
    # # chunked_prefill_p99_itls = []
    # # chunked_prefill_slos = []

    # # for chunk_size in [512, 1024, 2048, 4096]:
    # #     filter = (df["dataset_name"] == benchmark_dataset) & (df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == chunk_size) & (df["request_rate"] == request_rate) & (df["model_name"] == model_name)
    # #     chunked_prefill_ttfts = df[filter]["p99_ttft"].values.tolist()
    # #     chunked_prefill_itls = df[filter]["p99_itl"].values.tolist()
    # #     chunked_prefill_slo = df[filter]["slo"].values.tolist()
    # #     if len(chunked_prefill_ttfts) == 1:
    # #         chunked_prefill_p99_ttfts.append(chunked_prefill_ttfts[0])
    # #         chunked_prefill_p99_itls.append(chunked_prefill_itls[0])
    # #         chunked_prefill_slos.append(chunked_prefill_slo[0])
    # #     else:
    # #         print(f"chunk_size {chunk_size} has no or multiple entries: {chunked_prefill_ttfts}")

    # # print(chunked_prefill_p99_ttfts)
    # # print(chunked_prefill_p99_itls)
    # # print(chunked_prefill_slos)

    # # staged_prefill_p99_ttfts = []
    # # staged_prefill_p99_itls = []
    # # staged_prefill_slos = []

    # # for num_stage in [1, 2, 4, 8, 16]:
    # #     filter = (df["dataset_name"] == benchmark_dataset) & (df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == num_stage) & (df["request_rate"] == request_rate) & (df["model_name"] == model_name) & (df["max_num_batched_tokens"] == max_num_batched_tokens)
    # #     staged_prefill_ttfts = df[filter]["p99_ttft"].values.tolist()
    # #     staged_prefill_itls = df[filter]["p99_itl"].values.tolist()
    # #     staged_prefill_slo = df[filter]["slo"].values.tolist()
    # #     if len(staged_prefill_ttfts) == 1:
    # #         staged_prefill_p99_ttfts.append(staged_prefill_ttfts[0])
    # #         staged_prefill_p99_itls.append(staged_prefill_itls[0])
    # #         staged_prefill_slos.append(staged_prefill_slo[0])
    # #     else:
    # #         print(f"num_stage {num_stage} has no or multiple entries: {staged_prefill_ttfts}")

    # # print(staged_prefill_p99_ttfts)
    # # print(staged_prefill_p99_itls)
    # # print(staged_prefill_slos)

    # # plt.figure(figsize=(12, 6))
    # # plt.plot(chunked_prefill_p99_itls, chunked_prefill_p99_ttfts, marker='o', label='Chunked Prefill')
    # # plt.plot(staged_prefill_p99_itls, staged_prefill_p99_ttfts, marker='o', label='Staged Prefill')

    # # plt.xlabel('P99 ITL (s)')
    # # plt.ylabel('P99 TTFT (s)')
    # # plt.title(f'P99 TTFT vs P99 ITL for {model_name} on {benchmark_dataset} at request rate {request_rate}')
    # # plt.legend()
    # # plt.grid(True)
    # # plt.savefig(f"p99_ttft_vs_itl_{model_name.replace('--', '_')}_{benchmark_dataset}_request_rate_{request_rate}.pdf")

    # # 16384 1394.94 8192 1479.21 4096 1563.45 2048 1782.23 1024 2625.53 512 4413.58

    # chunk_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    # latency = [4413.58, 2625.53, 1782.23, 1563.45, 1479.21, 1394.94]
    # latency = [3404.28, 1974.99, 1757.37, 1652.51, 1607.98, 1591.23]
    # num_chunks = [16384 // chunk_size for chunk_size in chunk_sizes]

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
    for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B", "models--openai--gpt-oss-20b"]):
        short_model_name = "Qwen3-30B-A3B" if model_name == "models--Qwen--Qwen3-30B-A3B" else "GPT-OSS-20B"
        arxiv_figure_label = chr(ord('a') + model_idx)
        sharegpt_figure_label = chr(ord('a') + model_idx + 2)
    # for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B"]):
        staged_prefill_num_stages = 16 if model_name == "models--Qwen--Qwen3-30B-A3B" else 12

        # --- Arxiv ---
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
            graph_df[graph_df["schedule_mode"] == "staged-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])

        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
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
            # & (df["max_num_seqs"] == (128 if model_name == "models--Qwen--Qwen3-30B-A3B" else 64))
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 100)
        ]

        ax = sns.barplot(
            graph_df, x="request_rate", y="slo",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
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
            graph_df[graph_df["schedule_mode"] == "staged-prefill"].reset_index(drop=True).reset_index(),
            graph_df[graph_df["schedule_mode"] == "chunked-prefill"].reset_index(drop=True).reset_index()
        ])
        sns.lineplot(
            data=mean_requests_df, x="index", y="mean_requests",
            hue="schedule_mode", hue_order=["chunked-prefill", "staged-prefill"],
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
        if "staged-prefill" == label:
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

    for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B", "models--openai--gpt-oss-20b"]):
    # for model_idx, model_name in enumerate(["models--Qwen--Qwen3-30B-A3B"]):
        short_model_name = "Qwen3-30B-A3B" if model_name == "models--Qwen--Qwen3-30B-A3B" else "GPT-OSS-20B"
        arxiv_figure_label = chr(ord('a') + model_idx)
        sharegpt_figure_label = chr(ord('a') + model_idx + 2)
        staged_prefill_num_stages = 16 if model_name == "models--Qwen--Qwen3-30B-A3B" else 12
        # --- Arxiv ---
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
                "type": "Layered prefill TTFT" if row.schedule_mode == "staged-prefill" else "Chunked prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered prefill TBT" if row.schedule_mode == "staged-prefill" else "Chunked prefill TBT",
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
            # & (df["max_num_seqs"] == (64 if model_name == "models--Qwen--Qwen3-30B-A3B" else 64))
            & (
                ((df["schedule_mode"] == "staged-prefill") & (df["num_stages"] == staged_prefill_num_stages) & (df["max_num_batched_tokens"] == 8192))
                | ((df["schedule_mode"] == "chunked-prefill") & (df["max_num_batched_tokens"] == 512))
            )
            & (df["num_requests"] >= 100)
        ]

        data = []
        for row in graph_df.itertuples():
            data.append({
                "request_rate": row.request_rate,
                "value": row.ttft_slo_attain,
                "type": "Layered prefill TTFT" if row.schedule_mode == "staged-prefill" else "Chunked prefill TTFT",
                # "schedule_mode": row.schedule_mode,
            })
            data.append({
                "request_rate": row.request_rate,
                "value": row.itl_slo_attain,
                "type": "Layered prefill TBT" if row.schedule_mode == "staged-prefill" else "Chunked prefill TBT",
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

    staged_prefill_json_file = "logs_good/benchmark_models--Qwen--Qwen3-30B-A3B_8192_256_32768_0.85_2_False_debug_staged-prefill_16_arxiv_-1_None_1.3_780.json"
    chunked_prefill_json_file = "logs_good/benchmark_models--Qwen--Qwen3-30B-A3B_512_256_32768_0.85_2_False_debug_chunked-prefill_1_arxiv_-1_None_1.3_780.json"

    idx = 205
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
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(bottom=0.15, top=0.90, left=0.15, right=0.95)
    axs.plot(chunked_prefill_times, chunked_prefill_num_tokens, label='Chunked prefill', color="#E38E39")
    axs.plot(staged_prefill_times, staged_prefill_num_tokens, label='Layered prefill', color="#7D9BBC")

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
