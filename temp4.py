import json

def get_num_experts_loaded(filename: str) -> int:
    num_expert_load = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[MoE]"):
                data = json.loads(line[len("[MoE]"):].strip())
                num_expert_load += len(data)

    return num_expert_load

if __name__ == "__main__":
    # print(f"{get_num_experts_loaded('chunked_prefill_512.log')}")
    # print(f"{get_num_experts_loaded('chunked_prefill_1024.log')}")
    # print(f"{get_num_experts_loaded('staged_prefill_8192.log')}")
    # print(f"{get_num_experts_loaded('chunked_prefill_512_0.5.log')}")
    # print(f"{get_num_experts_loaded('staged_prefill_8192_0.5.log')}")
    # print(f"{get_num_experts_loaded('chunked_prefill_512_0.3.log')}")
    # print(f"{get_num_experts_loaded('staged_prefill_8192_0.3.log')}")
    print(f"{get_num_experts_loaded('chunked_prefill_512_1.0.log')}")
    print(f"{get_num_experts_loaded('staged_prefill_8192_1.0.log')}")
    print(f"{get_num_experts_loaded('chunked_prefill_512_1.0_arxiv.log')}")
    print(f"{get_num_experts_loaded('staged_prefill_8192_1.0_arxiv.log')}")
    print(f"{get_num_experts_loaded('chunked_prefill_512_1.0_longbench.log')}")
    print(f"{get_num_experts_loaded('staged_prefill_8192_1.0_longbench.log')}")
