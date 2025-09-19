import pickle
import numpy as np
import torch
import torch.distributed as dist
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


class ModelRunner:
    """
    모델 실행을 담당하는 클래스
    Tensor Parallelism을 지원하며, 분산 환경에서 모델 추론을 수행
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        ModelRunner 초기화

        Args:
            config: 설정 객체
            rank: 현재 프로세스의 랭크 (0: 메인 프로세스, 1~: 워커 프로세스)
            event: 프로세스 간 통신을 위한 이벤트 객체
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KV 캐시 블록 크기
        self.enforce_eager = config.enforce_eager     # Eager 모드 강제 여부
        self.world_size = config.tensor_parallel_size # 전체 프로세스 수
        self.rank = rank                              # 현재 프로세스 랭크
        self.event = event                            # 프로세스 간 통신 이벤트

        # NCCL을 사용한 분산 통신 초기화
        dist.init_process_group("nccl", f"tcp://localhost:{config.nccl_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 현재 프로세스의 CUDA 디바이스 설정

        # 기본 dtype과 디바이스 설정 (모델 로딩을 위해)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype if hf_config.torch_dtype else torch.bfloat16)
        torch.set_default_device("cuda")

        # 모델 아키텍처에 따라 적절한 모델 클래스 선택
        if hf_config.architectures[0] == "Qwen3ForCausalLM":
            self.model = Qwen3ForCausalLM(hf_config)
        elif hf_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.model = Qwen3MoeForCausalLM(hf_config)
        elif hf_config.architectures[0] == "GptOssForCausalLM":
            self.model = GptOssForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model architecture: {hf_config.architectures[0]}")

        # 모델 가중치 로딩
        load_model(self.model, config.model)
        self.sampler = Sampler()  # 토큰 샘플링을 위한 샘플러

        self.intermediate_outputs = dict()

        # 모델 워밍업 및 KV 캐시 할당
        self.warmup_model()
        self.allocate_kv_cache()

        # Eager 모드가 아닌 경우 CUDA 그래프 캡처
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 기본 설정 복원
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Tensor Parallelism이 활성화된 경우 공유 메모리 설정
        if self.world_size > 1:
            if rank == 0:
                # 메인 프로세스: 공유 메모리 생성
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 모든 프로세스가 동기화될 때까지 대기
            else:
                # 워커 프로세스: 공유 메모리 연결 및 루프 시작
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """
        ModelRunner 종료 및 리소스 정리
        """
        if self.world_size > 1:
            # 공유 메모리 정리
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 공유 메모리 삭제

        if not self.enforce_eager:
            # CUDA 그래프 정리
            del self.graphs, self.graph_pool

        torch.cuda.synchronize()  # CUDA 작업 완료 대기
        dist.destroy_process_group()  # 분산 프로세스 그룹 정리

    def loop(self):
        """
        워커 프로세스의 메인 루프
        메인 프로세스로부터 명령을 받아 실행
        """
        while True:
            method_name, args = self.read_shm()  # 공유 메모리에서 명령 읽기
            self.call(method_name, *args)        # 명령 실행
            if method_name == "exit":            # 종료 명령이면 루프 종료
                break

    def read_shm(self):
        """
        공유 메모리에서 메서드 호출 정보 읽기

        Returns:
            tuple: (메서드명, 인자들)
        """
        assert self.world_size > 1 and self.rank  # 워커 프로세스에서만 호출 가능
        self.event.wait()  # 메인 프로세스의 신호 대기

        # 데이터 크기 읽기 (처음 4바이트)
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 실제 데이터 읽기 및 역직렬화
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 이벤트 초기화
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        공유 메모리에 메서드 호출 정보 쓰기

        Args:
            method_name: 호출할 메서드명
            *args: 메서드 인자들
        """
        assert self.world_size > 1 and not self.rank  # 메인 프로세스에서만 호출 가능
        data = pickle.dumps([method_name, *args])  # 데이터 직렬화
        n = len(data)

        # 데이터 크기와 실제 데이터를 공유 메모리에 쓰기
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data

        # 모든 워커 프로세스에 신호 전송
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        메서드 호출을 처리하는 래퍼

        Args:
            method_name: 호출할 메서드명
            *args: 메서드 인자들

        Returns:
            메서드 실행 결과
        """
        if self.world_size > 1 and self.rank == 0:
            # 메인 프로세스: 워커 프로세스들에게 명령 전송
            self.write_shm(method_name, *args)

        # 실제 메서드 호출
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        모델 워밍업: 첫 번째 추론을 위한 준비
        CUDA 커널 컴파일 및 메모리 할당을 미리 수행
        """
        torch.cuda.empty_cache()  # CUDA 캐시 정리
        torch.cuda.reset_peak_memory_stats()  # 메모리 통계 초기화

        # 최대 배치 크기와 모델 길이 계산
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_lens = []

        # 시퀀스 길이들을 계산 (최대 시퀀스 수까지)
        for _ in range(self.config.max_num_seqs):
            seq_len = min(max_num_batched_tokens, max_model_len)
            seq_lens.append(seq_len)
            max_num_batched_tokens -= seq_len
            if max_num_batched_tokens <= 0:
                break

        # 더미 시퀀스들 생성 및 워밍업 실행
        seqs = [Sequence([0] * seq_len) for seq_len in seq_lens]
        for seq in seqs:
            seq.status = SequenceStatus.PREFILLING
            seq.num_tokens_to_process = seq.num_prompt_tokens
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        KV 캐시 메모리 할당
        GPU 메모리를 고려하여 적절한 크기의 KV 캐시를 할당
        """
        config = self.config
        hf_config = config.hf_config

        # GPU 메모리 정보 수집
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # KV 캐시 블록당 필요한 메모리 계산
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize

        # 사용 가능한 메모리를 고려하여 블록 수 계산
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # KV 캐시 텐서 생성 [2, num_layers, num_blocks, block_size, num_heads, head_dim]
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)

        # 각 레이어에 KV 캐시 할당
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # Key 캐시
                module.v_cache = self.kv_cache[1, layer_id]  # Value 캐시
                layer_id += 1
        self.num_layers = layer_id

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        블록 테이블 준비
        각 시퀀스의 메모리 블록 할당 정보를 텐서로 변환

        Args:
            seqs: 시퀀스 리스트

        Returns:
            블록 테이블 텐서 또는 None
        """
        if not seqs:
            return None

        # 최대 블록 테이블 길이 계산
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0

        # 모든 시퀀스의 블록 테이블을 동일한 길이로 패딩
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare(self, seqs: list[Sequence]):
        """
        모델 실행을 위한 입력 데이터 준비
        Staged-Prefill 모드에서는 단계별로 다른 처리가 필요
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # Cumulative sequence lengths for query
        cu_seqlens_k = [0]  # Cumulative sequence lengths for key
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []   # KV 캐시 슬롯 매핑
        context_lens = []   # 컨텍스트 길이

        prefill_seqs = []      # PREFILLING 상태의 시퀀스들
        decode_seqs = []       # DECODING 상태의 시퀀스들
        prefill_compute_layers = []  # Staged-Prefill에서 계산할 레이어들
        inter_hidden_states = []     # 중간 hidden states
        inter_residual = []          # 중간 residual

        for seq in seqs:
            if seq.status == SequenceStatus.PREFILLING:
                # PREFILLING 시퀀스 처리
                prefill_seqs.append(seq)
                seqlen = seq.num_processed_tokens + seq.num_tokens_to_process

                # 입력 토큰과 위치 정보 추가
                input_ids.extend(seq[seq.num_processed_tokens:seqlen])
                positions.extend(list(range(seq.num_processed_tokens, seqlen)))

                # 시퀀스 길이 정보 업데이트
                seqlen_q = seq.num_tokens_to_process
                seqlen_k = seqlen
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                if not seq.block_table:
                    continue

                # 슬롯 매핑 계산 (KV 캐시에서의 위치)
                seq_slot_mapping = []
                for i in range(seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    seq_slot_mapping.extend(list(range(start, end)))

                slot_mapping.extend(seq_slot_mapping[seq.num_processed_tokens:seq.num_processed_tokens + seq.num_tokens_to_process])

                # Staged-Prefill 모드에서 중간 출력 처리
                if seq.stage != -1:
                    # 현재 단계에서 계산할 레이어들 결정
                    # prefill_compute_layers.append(np.arange(self.num_layers).reshape(seq.num_stages, -1)[seq.stage].tolist())
                    prefill_compute_layers.append(np.array_split(np.arange(self.num_layers), seq.num_stages)[seq.stage].tolist())

                    # 이전 단계의 중간 출력 가져오기
                    # i_hidden_states, i_residual = seq.intermediate_outputs if seq.intermediate_outputs else (None, None)
                    i_hidden_states, i_residual = self.intermediate_outputs.get(seq.seq_id, (None, None))
                    if seq.seq_id in self.intermediate_outputs:
                        del self.intermediate_outputs[seq.seq_id]
                    if i_hidden_states is not None:
                        inter_hidden_states.append(i_hidden_states)
                    if i_residual is not None:
                        inter_residual.append(i_residual)

            elif seq.status == SequenceStatus.DECODING:
                # DECODING 시퀀스 처리
                decode_seqs.append(seq)
                input_ids.append(seq.last_token)  # 마지막 토큰만 입력으로 사용
                positions.append(len(seq))        # 현재 시퀀스 길이를 위치로 사용
                context_lens.append(len(seq))     # 컨텍스트 길이

                # 마지막 블록의 마지막 슬롯 위치 계산
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            else:
                raise ValueError(f"Invalid sequence status: {seq.status}")

        # PREFILLING 여부 확인
        is_prefill = len(prefill_seqs) > 0
        prefill_block_tables = None

        # Chunked prefill인 경우 블록 테이블 준비
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # chunked prefill
            prefill_block_tables = self.prepare_block_tables(prefill_seqs)

        decode_block_tables = self.prepare_block_tables(decode_seqs)

        # 모든 텐서를 GPU로 이동
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 중간 출력들을 연결
        inter_hidden_states = torch.cat(inter_hidden_states, dim=0) if inter_hidden_states else None
        inter_residual = torch.cat(inter_residual, dim=0) if inter_residual else None

        # 컨텍스트 설정
        set_context(
            is_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            prefill_block_tables,
            decode_block_tables,
            prefill_compute_layers,
        )
        return input_ids, positions, (inter_hidden_states, inter_residual)

    def prepare_sample(self, seqs: list[Sequence]):
        """
        샘플링을 위한 온도 파라미터 준비

        Args:
            seqs: 시퀀스 리스트

        Returns:
            온도 텐서
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_outputs = None):
        """
        실제 모델 실행

        Args:
            input_ids: 입력 토큰 ID들
            positions: 위치 정보
            intermediate_outputs: 중간 출력 (Staged-Prefill용)

        Returns:
            tuple: (hidden_states, residual)
        """
        context = get_context()

        # PREFILLING이거나 Eager 모드이거나 배치 크기가 큰 경우 일반 실행
        if (
            context.is_prefill
            or self.enforce_eager
            or input_ids.size(0) > 512
        ):
            hidden_states, residual = self.model(input_ids, positions, intermediate_outputs)
            return hidden_states, residual
        else:
            # CUDA 그래프를 사용한 최적화된 실행
            bs = input_ids.size(0)
            context = get_context()

            # 적절한 배치 크기의 그래프 선택
            try:
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            except StopIteration:
                raise ValueError(f"No suitable CUDA graph found for batch size {bs}. Available sizes: {self.graph_bs}")
            graph_vars = self.graph_vars

            # 그래프 변수 초기화 (출력 제외)
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()

            # 그래프 변수에 현재 입력 설정
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.decode_block_tables.size(1)] = context.decode_block_tables

            # 그래프 재실행
            graph.replay()
            return graph_vars["outputs"][:bs], None

    def run(self, seqs: list[Sequence]) -> list[int]:
        """
        모델 실행의 메인 메서드

        Args:
            seqs: 처리할 시퀀스 리스트

        Returns:
            생성된 토큰 ID 리스트
        """
        # 입력 데이터 준비
        input_ids, positions, intermediate_outputs = self.prepare(seqs)

        # 샘플링 파라미터 준비 (메인 프로세스에서만)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 모델 실행
        hidden_states, residual = self.run_model(input_ids, positions, intermediate_outputs)

        # Staged-Prefill에서 중간 출력 저장
        start_idx = 0
        for seq in seqs:
            end_idx = start_idx + seq.num_tokens_to_process

            # Staged-Prefill 모드에서 중간 출력 저장
            if seq.stage != -1:
                # 현재 시퀀스에 해당하는 중간 출력 추출
                i_hidden_states = hidden_states
                if hidden_states is not None:
                    i_hidden_states = hidden_states[start_idx:end_idx]
                i_residual = residual
                if residual is not None:
                    i_residual = residual[start_idx:end_idx]
                # 시퀀스에 중간 출력 저장 (다음 단계에서 재사용)
                # seq.intermediate_outputs = (i_hidden_states, i_residual)
                self.intermediate_outputs[seq.seq_id] = (i_hidden_states, i_residual)

            # 마지막 단계 완료 후 중간 출력 삭제 (메모리 절약)
            if seq.stage >= seq.num_stages - 1:
                # seq.intermediate_outputs = None
                if seq.seq_id in self.intermediate_outputs:
                    del self.intermediate_outputs[seq.seq_id]
                # torch.cuda.empty_cache()

            start_idx = end_idx

        # 로짓 계산 및 토큰 샘플링
        with torch.inference_mode():
            logits = self.model.compute_logits(hidden_states)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 컨텍스트 정리
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        CUDA 그래프 캡처
        다양한 배치 크기에 대해 CUDA 그래프를 미리 캡처하여 추론 성능 최적화
        """
        config = self.config
        hf_config = config.hf_config

        # 최대 배치 크기 설정
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 그래프 캡처용 더미 텐서들 생성
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 캡처할 배치 크기들 정의
        self.graph_bs = list(range(1, 32)) + list(range(32, max_bs + 1, 8))
        self.graphs = {}
        self.graph_pool = None

        # 각 배치 크기에 대해 그래프 캡처
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # 컨텍스트 설정 (DECODING 모드)
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], decode_block_tables=block_tables[:bs])

            # 워밍업 실행
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])[0]

            # 그래프 캡처
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])[0]

            # 첫 번째 그래프의 풀을 다른 그래프들과 공유
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 그래프 변수들을 딕셔너리로 저장
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        if hasattr(self.model, "capture_cudagraph") and config.schedule_mode == "staged-prefill":
            # 모델이 CUDA 그래프 캡처를 지원하는 경우 추가 캡처
            self.model.capture_cudagraph(max_bs=max_bs, max_num_blocks=max_num_blocks, num_stages=config.num_stages)

        if hasattr(self.model, "capture_cudagraph_layers") and config.schedule_mode != "staged-prefill":
            # 모델 레이어별 CUDA 그래프 캡처
            self.model.capture_cudagraph_layers(max_num_batched_tokens=config.max_num_batched_tokens)
