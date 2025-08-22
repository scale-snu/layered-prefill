from collections import defaultdict
from typing import Iterable, Optional, Any

import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig


from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.utils.context import get_context, set_context, reset_context


class GptOssAttention(nn.Module):

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
        attention_bias: bool = True,
    ):
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

        self.sinks = torch.nn.Parameter(
            torch.empty(self.total_num_heads // tp_size,
                        dtype=torch.bfloat16,
                        requires_grad=False))

        self.qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.num_kv_heads,
            bias=attention_bias,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling={
                "type":
                "yarn",
                "factor":
                rope_scaling["factor"],
                "original_max_position_embeddings":
                rope_scaling["original_max_position_embeddings"],
                "beta_fast":
                rope_scaling["beta_fast"],
                "beta_slow":
                rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v, self.sinks.data)
        output = self.o_proj(attn_output)

        return output


class GptOssMLP(torch.nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        # Tensor Parallelism 크기가 전문가 수보다 클 수 없음
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.router = torch.nn.Linear(config.hidden_size,
                                      config.num_local_experts,
                                      dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(num_experts=config.num_local_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=False,
                                renormalize=True,
                                has_bias=True,
                                activation="swigluoai")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.router(x)
        t = self.experts(hidden_states=x, router_logits=g)
        if self.tp_size > 1:
            t = self.experts.maybe_all_reduce_tensor_model_parallel(t)
        return x


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
        )
        self.mlp = GptOssMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, 0


class GptOssModel(nn.Module):

    def __init__(
        self,
        *,
        config: GptOssConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = torch.nn.ModuleList([
            GptOssDecoderLayer(
                config,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
                nbs = next(x for x in self.graph_bs if x >= bs)
                if (pre_layers, nbs) in self.pre_graphs:
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
                nbs = next(x for x in self.graph_bs if x >= bs)
                if (post_layers, nbs) in self.post_graphs:
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


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv", "q"),
        "k_proj": ("qkv", "k"),
        "v_proj": ("qkv", "v"),
        # "gate_proj": ("gate_up_proj", 0),
        # "up_proj": ("gate_up_proj", 1),

        # # MoE 관련 매핑
        # "expert_gate_proj": ("expert_gate_up_proj", 0),
        # "expert_up_proj": ("expert_gate_up_proj", 1),
    }

    hf_to_vllm_mapper = {
        "orig_to_new_substr": {
            ".self_attn.": ".attn.",
            ".post_attention_layernorm.": ".mlp.norm.",
        },
        "orig_to_new_suffix": {
            ".embed_tokens.weight": ".embed_tokens.weight",
            ".input_layernorm.weight": ".attn.norm.weight",
            ".post_attention_layernorm.weight": ".mlp.norm.weight",

            # MoE MXFP4 weights
            ".gate_up_proj_blocks": ".w13_weight",
            ".down_proj_blocks": ".w2_weight",
            ".gate_up_proj_scales": ".w13_weight_scale",
            ".down_proj_scales": ".w2_weight_scale",

            # MoE other weights
            ".gate_up_proj": ".w13_weight",
            ".down_proj": ".w2_weight",

            # MoE Bias
            ".gate_up_proj_bias": ".w13_bias",
            ".down_proj_bias": ".w2_bias",
        },
    }

    def __init__(
        self,
        config: GptOssConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        temp_prefix = "model" if not prefix else f"{prefix}.model"

        self.model = GptOssModel(
            config=config,
            prefix=temp_prefix,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )

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
