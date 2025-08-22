import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig
import numpy as np
from typing import Callable, Literal, Optional, Union, overload, Any
from collections import defaultdict

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.utils.context import get_context, set_context, reset_context


class Qwen3MoeMLP(nn.Module):
    """
    Qwen3MoE 모델의 일반 MLP 레이어

    MoE가 아닌 일반적인 2-layer MLP입니다.
    SwiGLU 활성화 함수를 사용합니다.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        Qwen3MoeMLP 초기화

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
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False)

        # 활성화 함수 검증 및 설정
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
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
        x, _ = self.down_proj(x)        # Down 프로젝션 (Tensor Parallelism 고려)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Qwen3MoE 모델의 Sparse MoE 블록

    Mixture of Experts (MoE)를 구현합니다.
    여러 전문가(expert) 중에서 top-k개를 선택하여 계산합니다.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ):
        """
        Qwen3MoeSparseMoeBlock 초기화

        Args:
            config: Qwen3MoE 설정 객체
        """
        super().__init__()
        self.tp_size = dist.get_world_size()  # Tensor Parallelism 크기

        # Tensor Parallelism 크기가 전문가 수보다 클 수 없음
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        # MoE 설정
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok  # 토큰당 선택할 전문가 수
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        # ===== 핵심: Fused MoE 사용 =====
        # 최적화된 Fused MoE 구현을 사용하여 성능 향상
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,  # 결과를 병합하지 않음 (나중에 처리)
            renormalize=config.norm_topk_prob  # top-k 확률 재정규화
        )

        # Router (게이트) - 전문가 선택을 담당
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE 순전파

        Args:
            hidden_states: 입력 히든 상태 (1D 또는 2D 형태 가능)

        Returns:
            MoE 출력
        """
        # 원본 형태 저장
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)  # 2D로 변환

        # Router를 통한 전문가 선택 확률 계산
        router_logits = self.gate(hidden_states)

        # Fused MoE를 통한 전문가 계산
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits
        )

        # Tensor Parallelism이 활성화된 경우 All-Reduce 수행
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        # 원본 형태로 복원
        return final_hidden_states.view(orig_shape)


class Qwen3MoeAttention(nn.Module):
    """
    Qwen3MoE 모델의 어텐션 레이어

    Qwen3와 동일한 구조이지만 MoE 모델에 맞게 조정되었습니다.
    Tensor Parallelism을 지원하며, QK-Norm과 RoPE를 포함합니다.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
    ) -> None:
        """
        Qwen3MoeAttention 초기화

        Args:
            hidden_size: 히든 상태 차원
            num_heads: 어텐션 헤드 수
            num_kv_heads: Key-Value 헤드 수 (GQA/MQA 지원)
            rope_theta: RoPE theta 값
            rope_scaling: RoPE 스케일링 설정
            max_position_embeddings: 최대 위치 임베딩 길이
            head_dim: 각 헤드의 차원 (None이면 자동 계산)
            rms_norm_eps: RMS 정규화 epsilon
            qkv_bias: QKV 프로젝션에 bias 사용 여부
        """
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size()  # Tensor Parallelism 크기

        # 헤드 수 설정 (Tensor Parallelism 고려)
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads

        # KV 헤드 수가 TP 크기보다 큰 경우: KV 헤드를 여러 GPU에 분할
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV 헤드 수가 TP 크기보다 작은 경우: KV 헤드를 여러 GPU에 복제
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # 헤드 차원 및 크기 계산
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5  # 어텐션 스케일링 팩터
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

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
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
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
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # RoPE 적용
        q, k = self.rotary_emb(positions, q, k)

        # 어텐션 계산
        attn_output = self.attn(q, k, v)

        # 출력 프로젝션
        output = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    """
    Qwen3MoE 모델의 디코더 레이어

    Self-Attention과 MoE/MLP를 포함하며, Pre-LayerNorm 구조를 사용합니다.
    Staged-Prefill 모드에서 단계별 처리를 지원합니다.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        prefix: str = "",
    ) -> None:
        """
        Qwen3MoeDecoderLayer 초기화

        Args:
            config: Qwen3MoE 설정 객체
            prefix: 레이어 이름 접두사 (MoE 설정용)
        """
        super().__init__()

        self.is_graph_captured = False

        self.hidden_size = config.hidden_size

        # Self-Attention 레이어
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings",8192),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # ===== MoE 또는 일반 MLP 선택 =====
        # 레이어 이름에서 레이어 인덱스 추출
        subnames = prefix.split(".")
        int_vals: list[int] = []
        for subname in subnames:
            try:
                int_vals.append(int(subname))
            except ValueError:
                continue
        assert len(int_vals) == 1, (f"layer name {prefix} should"
                                    " only contain one integer")
        layer_idx = int_vals[0]

        # MLP만 사용하는 레이어 목록 확인
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)

        # MoE 사용 조건:
        # 1. MLP-only 레이어가 아님
        # 2. 전문가 수가 0보다 큼
        # 3. decoder_sparse_step 간격에 해당
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            # MoE 블록 사용
            self.mlp = Qwen3MoeSparseMoeBlock(config=config)
        else:
            # 일반 MLP 사용
            self.mlp = Qwen3MoeMLP(
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

        bs = hidden_states.size(0)
        if self.is_graph_captured and bs <= max(self.graph_bs):
            # CUDA 그래프가 캡처된 경우
            pre_graph = self.pre_graphs[next(x for x in self.graph_bs if x >= bs)]
            post_graph = self.post_graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            for k, v in graph_vars.items():
                if k.startswith("outputs_"):
                    v.zero_()

            graph_vars["hidden_states"][:bs] = hidden_states
            if residual is not None:
                graph_vars["residual"][:bs] = residual
            graph_vars["positions"][:bs] = positions

            pre_graph.replay()

            q = graph_vars["outputs_q"][:bs]
            k = graph_vars["outputs_k"][:bs]
            v = graph_vars["outputs_v"][:bs]

            attn_o = self.self_attn.attn(q, k, v)

            graph_vars["attn_o"][:bs] = attn_o
            graph_vars["residual"][:bs] = graph_vars["outputs_residual"][:bs]

            for k, v in graph_vars.items():
                if k.startswith("outputs_"):
                    v.zero_()

            post_graph.replay()

            hidden_states = graph_vars["outputs_hidden_states"][:bs].clone()
            residual = graph_vars["outputs_residual"][:bs].clone()

            return hidden_states, residual
        else:
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

    def _pre_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        # Pre-LayerNorm 구조
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        qkv = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)

        # QK-Norm 적용
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.self_attn.head_dim, self.self_attn.head_dim)
        q_by_head = self.self_attn.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.self_attn.head_dim, self.self_attn.head_dim)
        k_by_head = self.self_attn.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        # RoPE 적용
        q, k = self.self_attn.rotary_emb(positions, q, k)

        return residual, q, k, v

    def _post_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        hidden_states = self.self_attn.o_proj(hidden_states)

        # Post-Attention LayerNorm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


    def capture_cudagraph(self, layer_idx: int = 0, max_num_batched_tokens: int = 1024, graph_vars: dict = None):
        self.is_graph_captured = True

        self.graph_vars = graph_vars
        hidden_states = graph_vars["hidden_states"]
        residual = graph_vars["residual"]
        positions = graph_vars["positions"]
        attn_o = graph_vars["attn_o"]
        outputs_hidden_states = graph_vars["outputs_hidden_states"]
        outputs_residual = graph_vars["outputs_residual"]
        outputs_q = graph_vars["outputs_q"]
        outputs_k = graph_vars["outputs_k"]
        outputs_v = graph_vars["outputs_v"]

        self.graph_bs = []
        bs = 512
        while bs <= min(max_num_batched_tokens, 512):
            self.graph_bs.append(bs)
            bs = bs * 2
        self.pre_graphs = {}
        self.post_graphs = {}
        self.pre_graph_pool = None
        self.post_graph_pool = None

        for bs in reversed(self.graph_bs):
            pre_graph = torch.cuda.CUDAGraph()

            _positions = positions[:bs].clone()
            _hidden_states = hidden_states[:bs].clone()
            if layer_idx == 0:
                _residual = None
            else:
                _residual = residual[:bs].clone()

            _ = self._pre_forward(_positions, _hidden_states, _residual)

            # 그래프 캡쳐
            with torch.cuda.graph(pre_graph, self.pre_graph_pool):
                _positions = positions[:bs].clone()
                _hidden_states = hidden_states[:bs].clone()
                if layer_idx == 0:
                    _residual = None
                else:
                    _residual = residual[:bs].clone()

                _residual, _q, _k, _v = self._pre_forward(_positions, _hidden_states, _residual)

                # 출력 저장
                outputs_q[:bs] = _q
                outputs_k[:bs] = _k
                outputs_v[:bs] = _v
                outputs_residual[:bs] = _residual

            post_graph = torch.cuda.CUDAGraph()

            _positions = positions[:bs].clone()
            _attn_o = attn_o[:bs].clone()
            _residual = residual[:bs].clone()

            _ = self._post_forward(_attn_o, _residual)

            with torch.cuda.graph(post_graph, self.post_graph_pool):
                _positions = positions[:bs].clone()
                _attn_o = attn_o[:bs].clone()
                _residual = residual[:bs].clone()

                _hidden_states, _residual = self._post_forward(_attn_o, _residual)

                outputs_hidden_states[:bs] = _hidden_states
                outputs_residual[:bs] = _residual

            # 그래프 저장
            if self.pre_graph_pool is None:
                self.pre_graph_pool = pre_graph.pool()
            if self.post_graph_pool is None:
                self.post_graph_pool = post_graph.pool()

            self.pre_graphs[bs] = pre_graph
            self.post_graphs[bs] = post_graph

            torch.cuda.synchronize()

        print(f"Captured {len(self.pre_graphs) + len(self.post_graphs)} CUDA graphs for layer {layer_idx}.")


class Qwen3MoeModel(nn.Module):
    """
    Qwen3MoE 모델의 메인 클래스

    토큰 임베딩, 디코더 레이어들, 그리고 최종 정규화를 포함합니다.
    Staged-Prefill 모드에서 단계별 레이어 계산을 지원합니다.
    """

    def __init__(self, *, config: Qwen3MoeConfig, prefix: str = ""):
        """
        Qwen3MoeModel 초기화

        Args:
            config: Qwen3MoE 설정 객체
            prefix: 모델 이름 접두사
        """
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        # 토큰 임베딩 (Vocabulary Parallelism 지원)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # 디코더 레이어들 (현재는 Pipeline Parallelism 미지원)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config=config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])

        # 최종 정규화
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_graph_captured = False
        self.num_stages = -1

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """입력 임베딩 반환"""
        return self.embed_tokens(input_ids)

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

        # ===== STAGED-PREFILL 모드 처리 =====
        if context.prefill_compute_layers:
            residual = hidden_states.clone()

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
            prefill_compute_layer_map = defaultdict(list)
            for idx, prefill_compute_layer in enumerate(context.prefill_compute_layers):
                for layer_idx in prefill_compute_layer:
                    prefill_compute_layer_map[layer_idx].append(idx)

            # 단계별로 레이어를 나누기

            pre_layers = tuple(list(np.arange(min(prefill_compute_layer_map))))
            post_layers = tuple(list(np.arange(max(prefill_compute_layer_map) + 1, len(self.layers))))

            bs = hidden_states[len_staged_prefill:].size(0)
            if bs > 0:
                nbs = next(x for x in self.graph_bs if x >= bs) if self.is_graph_captured else -1
                if self.is_graph_captured and (pre_layers, nbs) in self.pre_graphs:
                    graph = self.pre_graphs[(pre_layers, nbs)]
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
                    for layer_idx in pre_layers:
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
                            _hidden_states, _residual = self.layers[layer_idx](positions[len_staged_prefill:], hidden_states[len_staged_prefill:], None)
                        else:
                            _hidden_states, _residual = self.layers[layer_idx](
                                positions[len_staged_prefill:],
                                hidden_states[len_staged_prefill:],
                                residual[len_staged_prefill:]
                            )
                        hidden_states[len_staged_prefill:] = _hidden_states
                        residual[len_staged_prefill:] = _residual

                        # 컨텍스트 복원
                        context.is_prefill = is_prefill
                        context.slot_mapping = slot_mapping

            # ===== 단계별 PREFILLING 모드 처리 =====

            for layer_idx in sorted(sorted(prefill_compute_layer_map)):
                layer = self.layers[layer_idx]
                # 현재 단계에서 계산할 레이어
                prefill_seq_indices = prefill_compute_layer_map[layer_idx]

                if prefill_compute_layer_map.get(layer_idx - 1) != prefill_seq_indices:
                    indices = []
                    for prefill_seq_idx in prefill_seq_indices:
                        start_idx = context.cu_seqlens_q[prefill_seq_idx]
                        end_idx = context.cu_seqlens_q[prefill_seq_idx + 1]
                        indices.append(
                            torch.arange(start_idx, end_idx, device=hidden_states.device)
                        )
                    indices.append(
                        torch.arange(len_staged_prefill, hidden_states.size(0), device=hidden_states.device)
                    )
                    indices = torch.cat(indices)

                    # 컨텍스트 정보 백업 및 수정
                    max_seqlen_k = context.max_seqlen_k
                    max_seqlen_q = context.max_seqlen_q
                    context.max_seqlen_k = -1
                    context.max_seqlen_q = -1
                    for prefill_seq_idx in prefill_seq_indices:
                        context.max_seqlen_k = max(context.max_seqlen_k, context.cu_seqlens_k[prefill_seq_idx + 1] - context.cu_seqlens_k[prefill_seq_idx])
                        context.max_seqlen_q = max(context.max_seqlen_q, context.cu_seqlens_q[prefill_seq_idx + 1] - context.cu_seqlens_q[prefill_seq_idx])

                    # 시퀀스 길이 정보 수정
                    cu_seqlens_k = context.cu_seqlens_k.clone()
                    cu_seqlens_q = context.cu_seqlens_q.clone()
                    context.cu_seqlens_k = [0]
                    context.cu_seqlens_q = [0]
                    for prefill_seq_idx in prefill_seq_indices:
                        context.cu_seqlens_k.append(cu_seqlens_k[prefill_seq_idx + 1] - cu_seqlens_k[prefill_seq_idx])
                        context.cu_seqlens_q.append(cu_seqlens_q[prefill_seq_idx + 1] - cu_seqlens_q[prefill_seq_idx])
                    context.cu_seqlens_k = torch.tensor(context.cu_seqlens_k, device=hidden_states.device, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
                    context.cu_seqlens_q = torch.tensor(context.cu_seqlens_q, device=hidden_states.device, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)

                    # 블록 테이블 및 슬롯 매핑 수정
                    prefill_block_tables = None
                    if context.prefill_block_tables is not None:
                        prefill_block_tables = context.prefill_block_tables.clone()
                        context.prefill_block_tables = []
                        for prefill_seq_idx in prefill_seq_indices:
                            start_idx = cu_seqlens_q[prefill_seq_idx]
                            end_idx = cu_seqlens_q[prefill_seq_idx + 1]
                            context.prefill_block_tables.append(prefill_block_tables[start_idx:end_idx])
                        context.prefill_block_tables = torch.cat(context.prefill_block_tables, dim=0)

                    slot_mapping = None
                    if context.slot_mapping is not None:
                        slot_mapping = context.slot_mapping.clone()
                        context.slot_mapping = slot_mapping[indices]

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

            bs = hidden_states[len_staged_prefill:].size(0)
            if bs > 0:
                nbs = next(x for x in self.graph_bs if x >= bs) if self.is_graph_captured else -1
                if self.is_graph_captured and (post_layers, nbs) in self.post_graphs:
                    graph = self.post_graphs[(post_layers, nbs)]
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
                    for layer_idx in post_layers:
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
                            _hidden_states, _residual = self.layers[layer_idx](positions[len_staged_prefill:], hidden_states[len_staged_prefill:], None)
                        else:
                            _hidden_states, _residual = self.layers[layer_idx](
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
                indices = []
                for prefill_seq_idx in prefill_compute_layer_map[len(self.layers) - 1]:
                    start_idx = context.cu_seqlens_q[prefill_seq_idx]
                    end_idx = context.cu_seqlens_q[prefill_seq_idx + 1]
                    indices.append(torch.arange(start_idx, end_idx, device=hidden_states.device))
                indices.append(torch.arange(len_staged_prefill, hidden_states.size(0), device=hidden_states.device))
                indices = torch.cat(indices)
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
            residual = None
            # ===== 일반 모드 (Staged-Prefill 아님) =====
            # 모든 레이어를 순차적으로 처리
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)

            # 최종 정규화
            hidden_states, residual = self.norm(hidden_states, residual)

        return hidden_states, residual

    def capture_cudagraph_layers(self, max_num_batched_tokens: int = 1024):
        """
        CUDA 그래프 캡처

        Staged-Prefill 모드에서 각 레이어의 CUDA 그래프를 캡처합니다.
        """

        hidden_size = self.embed_tokens.weight.size(1)

        hidden_states = torch.zeros(max_num_batched_tokens, hidden_size)
        residual = torch.zeros(max_num_batched_tokens, hidden_size)
        positions = torch.zeros(max_num_batched_tokens, dtype=torch.int64)
        attn_o = torch.zeros(max_num_batched_tokens, self.layers[0].self_attn.total_num_heads * self.layers[0].self_attn.head_dim)
        outputs_hidden_states = torch.zeros(max_num_batched_tokens, hidden_size)
        outputs_residual = torch.zeros(max_num_batched_tokens, hidden_size)
        outputs_q = torch.zeros(max_num_batched_tokens, self.layers[0].self_attn.q_size)
        outputs_k = torch.zeros(max_num_batched_tokens, self.layers[0].self_attn.kv_size)
        outputs_v = torch.zeros(max_num_batched_tokens, self.layers[0].self_attn.kv_size)

        graph_vars = dict(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            attn_o=attn_o,
            outputs_hidden_states=outputs_hidden_states,
            outputs_residual=outputs_residual,
            outputs_q=outputs_q,
            outputs_k=outputs_k,
            outputs_v=outputs_v,
        )

        for layer_idx, layer in enumerate(self.layers):
            layer.capture_cudagraph(layer_idx, max_num_batched_tokens, graph_vars)

    def capture_cudagraph(self, max_bs: int = 1, max_num_blocks: int = 1, num_stages: int = 1):
        """
        CUDA 그래프 캡처

        Staged-Prefill 모드에서 CUDA 그래프를 캡처합니다.
        """

        self.is_graph_captured = True

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

        self.graph_bs = list(range(1, 16)) + list(range(16, max_bs + 1, 8))
        self.pre_graphs = {}
        self.post_graphs = {}
        self.graph_pool = None

        for ith_stage in range(num_stages):
            pre_layer_idices = list(np.concatenate(np.array_split(np.arange(len(self.layers)), num_stages)[:ith_stage])) if ith_stage > 0 else []
            post_layer_idices = list(np.concatenate(np.array_split(np.arange(len(self.layers)), num_stages)[ith_stage:]))

            if len(pre_layer_idices) > 0:
                for bs in reversed(self.graph_bs):
                    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], decode_block_tables=block_tables[:bs])

                    graph = torch.cuda.CUDAGraph()

                    # 워밍업 실행
                    _positions = positions[:bs].clone()
                    _hidden_states = hidden_states[:bs].clone()
                    _residual = residual[:bs].clone()

                    for layer_idx in pre_layer_idices:
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
                        for layer_idx in pre_layer_idices:
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

                    self.pre_graphs[(tuple(pre_layer_idices), bs)] = graph

            if len(post_layer_idices) > 0:
                for bs in reversed(self.graph_bs):
                    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], decode_block_tables=block_tables[:bs])

                    graph = torch.cuda.CUDAGraph()

                    # 워밍업 실행
                    _positions = positions[:bs].clone()
                    _hidden_states = hidden_states[:bs].clone()
                    _residual = residual[:bs].clone()

                    for layer_idx in post_layer_idices:
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
                        for layer_idx in post_layer_idices:
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

                    self.post_graphs[(tuple(post_layer_idices), bs)] = graph

                print(ith_stage)

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

        print(f"Captured {len(self.pre_graphs) + len(self.post_graphs)} CUDA graphs for Staged-Prefill mode with {num_stages} stages.")


class Qwen3MoeForCausalLM(nn.Module):
    """
    Qwen3MoE 언어 모델 클래스

    토큰 생성과 로짓 계산을 담당합니다.
    Staged-Prefill 모드를 완전히 지원하며, MoE 구조를 포함합니다.
    """

    # 모듈 매핑 (가중치 로딩용)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),

        # MoE 관련 매핑
        "expert_gate_proj": ("expert_gate_up_proj", 0),
        "expert_up_proj": ("expert_gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig,
        prefix: str = ""
    ) -> None:
        """
        Qwen3MoeForCausalLM 초기화

        Args:
            config: Qwen3MoE 설정 객체
            prefix: 모델 이름 접두사
        """
        super().__init__()
        temp_prefix = "model" if not prefix else f"{prefix}.model"
        self.model = Qwen3MoeModel(config=config, prefix=temp_prefix)

        # 언어 모델 헤드 (Vocabulary Parallelism 지원)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 가중치 공유 (선택적)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """입력 임베딩 반환"""
        return self.model.get_input_embeddings(input_ids)

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
    ) -> Optional[torch.Tensor]:
        """
        로짓 계산

        Args:
            hidden_states: 히든 상태

        Returns:
            로짓 텐서
        """
        logits = self.lm_head(hidden_states)
        return logits

    def capture_cudagraph_layers(self, max_num_batched_tokens: int = 1024):
        """
        CUDA 그래프 캡처

        각 레이어의 CUDA 그래프를 캡처합니다.
        """
        self.model.capture_cudagraph_layers(max_num_batched_tokens)

    def capture_cudagraph(self, max_bs: int = 1, max_num_blocks: int = 1, num_stages: int = 1):
        """
        CUDA 그래프 캡처

        Staged-Prefill 모드에서 CUDA 그래프를 캡처합니다.
        """
        self.model.capture_cudagraph(max_bs, max_num_blocks, num_stages)
