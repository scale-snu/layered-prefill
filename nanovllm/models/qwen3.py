import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
import numpy as np

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context, set_context, reset_context


class Qwen3Attention(nn.Module):
    """
    Qwen3 모델의 어텐션 레이어

    Tensor Parallelism을 지원하며, QK-Norm과 RoPE를 포함합니다.
    Staged-Prefill 모드에서는 단계별로 다른 레이어들을 계산할 수 있습니다.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """
        Qwen3Attention 초기화

        Args:
            hidden_size: 히든 상태 차원
            num_heads: 어텐션 헤드 수
            num_kv_heads: Key-Value 헤드 수 (GQA/MQA 지원)
            max_position: 최대 위치 임베딩 길이
            head_dim: 각 헤드의 차원 (None이면 자동 계산)
            rms_norm_eps: RMS 정규화 epsilon
            qkv_bias: QKV 프로젝션에 bias 사용 여부
            rope_theta: RoPE theta 값
            rope_scaling: RoPE 스케일링 설정
        """
        super().__init__()
        tp_size = dist.get_world_size()  # Tensor Parallelism 크기

        # 헤드 수 설정 (Tensor Parallelism 고려)
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # 헤드 차원 및 크기 계산
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5  # 어텐션 스케일링 팩터

        # QKV 프로젝션 (병합된 선형 레이어)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # 출력 프로젝션
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # RoPE (Rotary Position Embedding) 설정
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # 어텐션 계산 (Flash Attention 사용)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # QK-Norm (Query-Key 정규화)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        어텐션 계산 수행

        Args:
            positions: 위치 정보 텐서
            hidden_states: 입력 히든 상태

        Returns:
            어텐션 출력
        """
        # QKV 프로젝션 및 분할
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-Norm 적용
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # RoPE 적용
        q, k = self.rotary_emb(positions, q, k)

        # 어텐션 계산
        o = self.attn(q, k, v)

        # 출력 프로젝션
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 모델의 MLP 레이어

    SwiGLU 활성화 함수를 사용하는 2-layer MLP입니다.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        Qwen3MLP 초기화

        Args:
            hidden_size: 입력/출력 히든 상태 차원
            intermediate_size: 중간 레이어 차원
            hidden_act: 활성화 함수 (현재는 "silu"만 지원)
        """
        super().__init__()

        # Gate와 Up 프로젝션을 병합한 선형 레이어
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate와 up을 위한 두 개의 출력
            bias=False,
        )

        # Down 프로젝션
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        # 활성화 함수 검증 및 설정
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()  # SwiGLU 활성화

    def forward(self, x):
        """
        MLP 순전파

        Args:
            x: 입력 텐서

        Returns:
            MLP 출력
        """
        gate_up = self.gate_up_proj(x)  # Gate와 Up 프로젝션
        x = self.act_fn(gate_up)        # SwiGLU 활성화
        x = self.down_proj(x)           # Down 프로젝션
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 모델의 디코더 레이어

    Self-Attention과 MLP를 포함하며, Pre-LayerNorm 구조를 사용합니다.
    Staged-Prefill 모드에서 단계별 처리를 지원합니다.
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        Qwen3DecoderLayer 초기화

        Args:
            config: Qwen3 설정 객체
        """
        super().__init__()

        # Self-Attention 레이어
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # MLP 레이어
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # Layer Normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        디코더 레이어 순전파

        Args:
            positions: 위치 정보
            hidden_states: 입력 히든 상태
            residual: 잔차 연결 (None이면 초기화)

        Returns:
            (hidden_states, residual): 출력 히든 상태와 잔차
        """
        # Pre-LayerNorm 구조
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self-Attention
        hidden_states = self.self_attn(positions, hidden_states)

        # Post-Attention LayerNorm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 모델의 메인 클래스

    토큰 임베딩, 디코더 레이어들, 그리고 최종 정규화를 포함합니다.
    Staged-Prefill 모드에서 단계별 레이어 계산을 지원합니다.
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        Qwen3Model 초기화

        Args:
            config: Qwen3 설정 객체
        """
        super().__init__()

        # 토큰 임베딩 (Vocabulary Parallelism 지원)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # 디코더 레이어들
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 최종 정규화
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.num_stages = -1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        모델 순전파

        Args:
            input_ids: 입력 토큰 ID들
            positions: 위치 정보
            intermediate_outputs: Staged-Prefill용 중간 출력 (hidden_states, residual)

        Returns:
            (hidden_states, residual): 최종 히든 상태와 잔차
        """
        context = get_context()

        # Staged-Prefill에서 PREFILLING 토큰 수 계산
        len_staged_prefill = context.cu_seqlens_q[-1] if context.is_prefill else 0

        # 토큰 임베딩 및 초기 잔차 설정
        hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states.clone()

        # ===== STAGED-PREFILL 모드 처리 =====
        if context.prefill_compute_layers:
            # 중간 출력이 있으면 재사용 (이전 단계의 결과)
            if intermediate_outputs is not None:
                i_hidden_states, i_residual = intermediate_outputs
                if i_hidden_states is not None:
                    # 이전 단계의 hidden_states와 현재 입력을 연결
                    hidden_states = torch.cat([i_hidden_states, hidden_states[i_hidden_states.size(0):]], dim=0)
                if i_residual is not None:
                    # 이전 단계의 residual과 현재 residual을 연결
                    residual = torch.cat([i_residual, residual[i_residual.size(0):]], dim=0)

            # 단계별 레이어 매핑 생성
            prefill_compute_layer_map = {}
            for idx, prefill_compute_layer in enumerate(context.prefill_compute_layers):
                for layer_idx in prefill_compute_layer:
                    prefill_compute_layer_map[layer_idx] = idx

            if self.num_stages == -1:
                slayers = [self.layers]
                layers_per_stage = len(self.layers)
            else:
                # 단계별로 레이어를 나누기
                slayers = np.array_split(self.layers, self.num_stages)
                layers_per_stage = len(slayers[0])

            # 각 레이어를 단계별로 처리
            for stage_idx, layers in enumerate(slayers):
                if self.num_stages == -1 or stage_idx * layers_per_stage in prefill_compute_layer_map:
                    for layer_idx, layer in enumerate(layers):
                        layer_idx += layers_per_stage * stage_idx
                        # 현재 단계에서 계산할 레이어
                        idx = prefill_compute_layer_map[layer_idx]

                        if prefill_compute_layer_map.get(layer_idx - 1, -1) != idx:
                            start_idx = context.cu_seqlens_q[idx]
                            end_idx = context.cu_seqlens_q[idx + 1]

                            # 처리할 인덱스 계산 (현재 단계 + 디코딩 부분)
                            indices = torch.cat([
                                torch.arange(start_idx, end_idx, device=hidden_states.device),
                                torch.arange(len_staged_prefill, hidden_states.size(0), device=hidden_states.device)
                            ])

                            # 컨텍스트 정보 백업 및 수정
                            max_seqlen_k = context.max_seqlen_k
                            max_seqlen_q = context.max_seqlen_q
                            context.max_seqlen_k = int(context.cu_seqlens_k[idx + 1] - context.cu_seqlens_k[idx])
                            context.max_seqlen_q = int(context.cu_seqlens_q[idx + 1] - context.cu_seqlens_q[idx])

                            # 시퀀스 길이 정보 수정
                            cu_seqlens_k = context.cu_seqlens_k.clone()
                            cu_seqlens_q = context.cu_seqlens_q.clone()
                            context.cu_seqlens_k = cu_seqlens_k[idx:idx + 2] - cu_seqlens_k[idx]
                            context.cu_seqlens_q = cu_seqlens_q[idx:idx + 2] - cu_seqlens_q[idx]

                            # 블록 테이블 및 슬롯 매핑 수정
                            prefill_block_tables = None
                            if context.prefill_block_tables is not None:
                                prefill_block_tables = context.prefill_block_tables.clone()
                                context.prefill_block_tables = prefill_block_tables[start_idx:end_idx]

                            slot_mapping = None
                            if context.slot_mapping is not None:
                                slot_mapping = context.slot_mapping.clone()
                                context.slot_mapping = context.slot_mapping[indices]

                            _positions = positions[indices]
                            _hidden_states = hidden_states[indices]
                            _residual = residual[indices] if layer_idx > 0 else None
                        # 레이어 실행
                        _hidden_states, _residual = layer(_positions, _hidden_states, _residual)

                        if prefill_compute_layer_map.get(layer_idx + 1, -1) != idx:
                            hidden_states[indices] = _hidden_states
                            residual[indices] = _residual

                            # 컨텍스트 정보 복원
                            context.max_seqlen_k = max_seqlen_k
                            context.max_seqlen_q = max_seqlen_q
                            context.cu_seqlens_k = cu_seqlens_k
                            context.cu_seqlens_q = cu_seqlens_q
                            context.prefill_block_tables = prefill_block_tables
                            context.slot_mapping = slot_mapping
                else:
                    bs = hidden_states[len_staged_prefill:].size(0)
                    if bs > 0:
                        if (stage_idx, bs) in self.graphs:
                            # CUDA 그래프가 있는 경우
                            graph = self.graphs[(stage_idx, bs)]
                            graph_vars = self.graph_vars

                            for k, v in graph_vars.items():
                                if not k.startswith("outputs_"):
                                    v.zero_()

                            graph_vars["hidden_states"][:bs] = hidden_states[len_staged_prefill:]
                            graph_vars["residual"][:bs] = residual[len_staged_prefill:]
                            graph_vars["positions"][:bs] = positions[len_staged_prefill:]
                            graph_vars["slot_mapping"][:bs] = context.slot_mapping[len_staged_prefill:]
                            graph_vars["context_lens"][:bs] = context.context_lens
                            graph_vars["block_tables"][:bs, :context.decode_block_tables.size(1)] = context.decode_block_tables

                            graph.replay()

                            hidden_states[len_staged_prefill:] = graph_vars["outputs_hidden_states"][:bs]
                            residual[len_staged_prefill:] = graph_vars["outputs_residual"][:bs]
                        else:
                            for layer_idx, layer in enumerate(layers):
                                # PREFILLING 모드를 일시적으로 비활성화
                                is_prefill = context.is_prefill
                                context.is_prefill = False

                                # 슬롯 매핑 수정
                                slot_mapping = None
                                if context.slot_mapping is not None:
                                    slot_mapping = context.slot_mapping.clone()
                                    context.slot_mapping = slot_mapping[len_staged_prefill:]

                                # 디코딩 부분만 레이어 실행
                                if layer_idx == 0:
                                    _hidden_states, _residual = layer(positions[len_staged_prefill:], hidden_states[len_staged_prefill:], None)
                                else:
                                    _hidden_states, _residual = layer(
                                        positions[len_staged_prefill:],
                                        hidden_states[len_staged_prefill:],
                                        residual[len_staged_prefill:]
                                    )
                                hidden_states[len_staged_prefill:] = _hidden_states
                                residual[len_staged_prefill:] = _residual

                                # 컨텍스트 복원
                                context.is_prefill = is_prefill
                                context.slot_mapping = slot_mapping

            # 최종 정규화 처리
            if len(self.layers) - 1 in prefill_compute_layer_map:
                # 마지막 레이어가 현재 단계에 포함된 경우
                idx = prefill_compute_layer_map[len(self.layers) - 1]
                start_idx = context.cu_seqlens_q[idx]
                end_idx = context.cu_seqlens_q[idx + 1]
                indices = torch.cat([
                    torch.arange(start_idx, end_idx, device=hidden_states.device),
                    torch.arange(len_staged_prefill, hidden_states.size(0), device=hidden_states.device)
                ])
                _hidden_states, _residual = self.norm(hidden_states[indices], residual[indices])
                hidden_states[indices] = _hidden_states
                residual[indices] = _residual
            else:
                # 마지막 레이어가 현재 단계에 포함되지 않은 경우 (디코딩 부분만)
                if hidden_states[len_staged_prefill:].size(0) > 0:
                    _hidden_states, _residual = self.norm(
                        hidden_states[len_staged_prefill:], residual[len_staged_prefill:]
                    )
                    hidden_states[len_staged_prefill:] = _hidden_states
                    residual[len_staged_prefill:] = _residual
        else:
            # ===== 일반 모드 (Staged-Prefill 아님) =====
            # 모든 레이어를 순차적으로 처리
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)

            # 최종 정규화
            hidden_states, residual = self.norm(hidden_states, residual)

        return hidden_states, residual

    def capture_cudagraph(self, max_bs: int = 1, max_num_blocks: int = 1, num_stages: int = 1):
        """
        CUDA 그래프 캡처

        Staged-Prefill 모드에서 CUDA 그래프를 캡처합니다.
        """

        self.num_stages = num_stages

        hidden_size = self.embed_tokens.weight.size(1)

        hidden_states = torch.zeros(max_bs, hidden_size)
        residual = torch.zeros(max_bs, hidden_size)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs_hidden_states = torch.zeros(max_bs, hidden_size)
        outputs_residual = torch.zeros(max_bs, hidden_size)

        self.graph_bs = list(range(1, 32)) + list(range(32, max_bs + 1, 8))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], decode_block_tables=block_tables[:bs])

            for ith_stage in range(num_stages):
                graph = torch.cuda.CUDAGraph()

                # 워밍업 실행
                _positions = positions[:bs].clone()
                _hidden_states = hidden_states[:bs].clone()
                _residual = residual[:bs].clone()
                for layer_idx in np.arange(len(self.layers)).reshape(num_stages, -1)[ith_stage]:
                    layer = self.layers[layer_idx]
                    if layer_idx == 0:
                        # 첫 번째 단계에서는 입력을 받음
                        _hidden_states, _residual = layer(_positions, _hidden_states, None)
                    else:
                        # 이후 단계에서는 이전 단계의 출력과 잔차를 사용
                        _hidden_states, _residual = layer(_positions, _hidden_states, _residual)

                outputs_hidden_states[:bs] = _hidden_states
                outputs_residual[:bs] = _residual

                # 그래프 캡쳐
                with torch.cuda.graph(graph, self.graph_pool):
                    _positions = positions[:bs].clone()
                    _hidden_states = hidden_states[:bs].clone()
                    _residual = residual[:bs].clone()
                    for layer_idx in np.arange(len(self.layers)).reshape(num_stages, -1)[ith_stage]:
                        layer = self.layers[layer_idx]
                        if layer_idx == 0:
                            # 첫 번째 단계에서는 입력을 받음
                            _hidden_states, _residual = layer(_positions, _hidden_states, None)
                        else:
                            # 이후 단계에서는 이전 단계의 출력과 잔차를 사용
                            _hidden_states, _residual = layer(_positions, _hidden_states, _residual)

                    outputs_hidden_states[:bs] = _hidden_states
                    outputs_residual[:bs] = _residual

                # 그래프 저장
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()

                self.graphs[(ith_stage, bs)] = graph
                torch.cuda.synchronize()

            reset_context()

        self.graph_vars = dict(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs_hidden_states=outputs_hidden_states,
            outputs_residual=outputs_residual,
        )

        print(f"Captured {len(self.graphs)} CUDA graphs for Staged-Prefill mode with {num_stages} stages.")


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 언어 모델 클래스

    토큰 생성과 로짓 계산을 담당합니다.
    Staged-Prefill 모드를 완전히 지원합니다.
    """

    # 모듈 매핑 (가중치 로딩용)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """
        Qwen3ForCausalLM 초기화

        Args:
            config: Qwen3 설정 객체
        """
        super().__init__()
        self.model = Qwen3Model(config)

        # 언어 모델 헤드 (Vocabulary Parallelism 지원)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 가중치 공유 (선택적)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        모델 순전파

        Args:
            input_ids: 입력 토큰 ID들
            positions: 위치 정보
            intermediate_outputs: Staged-Prefill용 중간 출력

        Returns:
            최종 히든 상태
        """
        hidden_states = self.model(input_ids, positions, intermediate_outputs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        로짓 계산

        Args:
            hidden_states: 히든 상태

        Returns:
            로짓 텐서
        """
        logits = self.lm_head(hidden_states)
        return logits

    def capture_cudagraph(self, max_bs: int = 1, max_num_blocks: int = 1, num_stages: int = 1):
        """
        CUDA 그래프 캡처

        Staged-Prefill 모드에서 CUDA 그래프를 캡처합니다.
        """
        self.model.capture_cudagraph(max_bs, max_num_blocks, num_stages)
