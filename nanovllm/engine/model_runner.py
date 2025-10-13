import gc
import pickle
import numpy as np
import torch
import torch.distributed as dist
import nvtx
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.models.gpt_oss import GptOssForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.layers.custom_all_reduce import capture
from nanovllm.utils.utils import disable_gc


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", f"tcp://localhost:{config.nccl_port}", world_size=self.world_size, rank=rank)

        torch.cuda.set_device(rank)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype if hf_config.torch_dtype else torch.bfloat16)
        torch.set_default_device("cuda")

        if hf_config.architectures[0] == "Qwen3ForCausalLM":
            self.model = Qwen3ForCausalLM(hf_config)
        elif hf_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.model = Qwen3MoeForCausalLM(hf_config)
        elif hf_config.architectures[0] == "GptOssForCausalLM":
            self.model = GptOssForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model architecture: {hf_config.architectures[0]}")

        load_model(self.model, config.model)
        self.sampler = Sampler()

        self.intermediate_outputs = dict()

        self.warmup_model()
        self.allocate_kv_cache()

        gc.collect()
        gc.freeze()

        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        gc.collect()
        gc.freeze()

        if self.world_size > 1:
            if rank == 0:
                try:
                    SharedMemory('nanovllm', create=False).unlink()
                except FileNotFoundError:
                    print("No existing shared memory to unlink.")
                    pass
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()

        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()

        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args], protocol=pickle.HIGHEST_PROTOCOL)
        n = len(data)

        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data

        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_lens = []

        for _ in range(self.config.max_num_seqs):
            seq_len = min(max_num_batched_tokens, max_model_len)
            seq_lens.append(seq_len)
            max_num_batched_tokens -= seq_len
            if max_num_batched_tokens <= 0:
                break

        seqs = [Sequence([0] * seq_len) for seq_len in seq_lens]
        for seq in seqs:
            seq.status = SequenceStatus.PREFILLING
            seq.num_tokens_to_process = seq.num_prompt_tokens
        self.run(seqs)

        seqs = [Sequence([0]) for seq_len in seq_lens]
        for seq in seqs:
            seq.status = SequenceStatus.PREFILLING
            seq.num_tokens_to_process = seq.num_prompt_tokens
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        self.num_layers = layer_id

    def prepare_block_tables(self, seqs: list[Sequence]):
        if not seqs:
            return None

        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0

        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    @nvtx.annotate("ModelRunner::prepare")
    def prepare(self, seqs: list[Sequence]):
        input_ids = np.array([], dtype=np.int64)
        positions = np.array([], dtype=np.int64)
        cu_seqlens_q = [0]  # Cumulative sequence lengths for query
        cu_seqlens_k = [0]  # Cumulative sequence lengths for key
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = []

        prefill_seqs = []
        decode_seqs = []
        prefill_compute_layers = None
        inter_hidden_states = []
        inter_residual = []
        inter_input_ids = []
        inter_positions = []

        for seq in seqs:
            if seq.status == SequenceStatus.PREFILLING:
                prefill_seqs.append(seq)
                seqlen = seq.num_processed_tokens + seq.num_tokens_to_process

                input_ids = np.concatenate((input_ids, seq[seq.num_processed_tokens:seqlen]))
                positions = np.concatenate((positions, np.arange(seq.num_processed_tokens, seqlen)))

                seqlen_q = seq.num_tokens_to_process
                seqlen_k = seqlen
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                if not seq.block_table:
                    continue

                seq_slot_mapping = []
                for i in range(seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    seq_slot_mapping.append(np.arange(start, end))
                seq_slot_mapping = np.concatenate(seq_slot_mapping)
                slot_mapping.extend(seq_slot_mapping[seq.num_processed_tokens:seq.num_processed_tokens + seq.num_tokens_to_process])

                if seq.stage != -1:
                    if prefill_compute_layers is not None:
                        if prefill_compute_layers != set(np.array_split(np.arange(self.num_layers), seq.num_stages)[seq.stage].tolist()):
                            raise ValueError("In Layered Prefill mode, all sequences must have the same stage.")
                    else:
                        prefill_compute_layers = set(np.array_split(np.arange(self.num_layers), seq.num_stages)[seq.stage].tolist())

                    i_hidden_states, i_residual, i_input_ids, i_positions = self.intermediate_outputs.get(seq.seq_id, (None, None, None, None))
                    if seq.seq_id in self.intermediate_outputs:
                        del self.intermediate_outputs[seq.seq_id]
                    if i_hidden_states is not None:
                        inter_hidden_states.append(i_hidden_states)
                    if i_residual is not None:
                        inter_residual.append(i_residual)
                    if i_input_ids is not None:
                        inter_input_ids.append(i_input_ids)
                    if i_positions is not None:
                        inter_positions.append(i_positions)

            elif seq.status == SequenceStatus.DECODING:
                decode_seqs.append(seq)
                input_ids = np.concatenate((input_ids, np.array([seq.last_token], dtype=np.int64)))
                positions = np.concatenate((positions, np.array([len(seq) - 1], dtype=np.int64)))
                context_lens.append(len(seq))

                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            else:
                raise ValueError(f"Invalid sequence status: {seq.status}")

        is_prefill = len(prefill_seqs) > 0
        prefill_block_tables = None

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # chunked prefill
            prefill_block_tables = self.prepare_block_tables(prefill_seqs)

        decode_block_tables = self.prepare_block_tables(decode_seqs)
        max_seqlen_k_dec = max(context_lens) if context_lens else 0

        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        inter_hidden_states = torch.cat(inter_hidden_states, dim=0) if inter_hidden_states else None
        inter_residual = torch.cat(inter_residual, dim=0) if inter_residual else None
        inter_input_ids = torch.cat(inter_input_ids, dim=0) if inter_input_ids else None
        inter_positions = torch.cat(inter_positions, dim=0) if inter_positions else None

        len_inter = inter_input_ids.size(0) if inter_input_ids is not None else 0
        input_ids = torch.from_numpy(input_ids[len_inter:]).cuda(non_blocking=True)
        positions = torch.from_numpy(positions[len_inter:]).cuda(non_blocking=True)
        input_ids = torch.cat([inter_input_ids, input_ids], dim=0) if inter_input_ids is not None else input_ids
        positions = torch.cat([inter_positions, positions], dim=0) if inter_positions is not None else positions

        set_context(
            is_prefill,
            int(cu_seqlens_q[-1]) if is_prefill else 0,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            max_seqlen_k_dec,
            slot_mapping,
            context_lens,
            prefill_block_tables,
            decode_block_tables,
            prefill_compute_layers,
        )
        return input_ids, positions, (inter_hidden_states, inter_residual)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @nvtx.annotate("ModelRunner::run_model")
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_outputs = None):
        hidden_states, residual = self.model(input_ids, positions, intermediate_outputs)
        return hidden_states, residual

    @nvtx.annotate("ModelRunner::run")
    @disable_gc()
    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions, intermediate_outputs = self.prepare(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        hidden_states, residual = self.run_model(input_ids, positions, intermediate_outputs)

        with nvtx.annotate("ModelRunner::store_intermediate"):
            start_idx = 0
            for seq in seqs:
                end_idx = start_idx + seq.num_tokens_to_process

                if seq.stage != -1 and seq.status == SequenceStatus.PREFILLING:
                    i_hidden_states = hidden_states
                    if hidden_states is not None:
                        i_hidden_states = hidden_states[start_idx:end_idx]
                    i_residual = residual
                    if residual is not None:
                        i_residual = residual[start_idx:end_idx]
                    i_input_ids = input_ids[start_idx:end_idx]
                    i_positions = positions[start_idx:end_idx]
                    self.intermediate_outputs[seq.seq_id] = (i_hidden_states, i_residual, i_input_ids, i_positions)

                if seq.stage >= seq.num_stages - 1:
                    if seq.seq_id in self.intermediate_outputs:
                        del self.intermediate_outputs[seq.seq_id]

                start_idx = end_idx

        with torch.inference_mode():
            logits = self.model.compute_logits(hidden_states)

        with nvtx.annotate("ModelRunner::sample"):
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        CUDA 그래프 캡처
        다양한 배치 크기에 대해 CUDA 그래프를 미리 캡처하여 추론 성능 최적화
        """
        config = self.config

        if hasattr(self.model, "capture_cudagraph_layers"):
            self.model.capture_cudagraph_layers(
                max_num_batched_tokens=config.max_num_batched_tokens,
                schedule_mode=config.schedule_mode,
            )
