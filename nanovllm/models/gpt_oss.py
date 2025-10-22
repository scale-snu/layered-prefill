from collections import defaultdict
from typing import Iterable, Optional, Any

import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig


from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.layers.custom_all_reduce import capture
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
        ith_layer: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
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
            self.total_num_kv_heads,
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
            window_size=128 if ith_layer % 2 == 0 else -1,
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
        return t


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, ith_layer: int = 0):
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
            ith_layer=ith_layer,
        )
        self.mlp = GptOssMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_graph_captured = False

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = hidden_states.size(0)
        if self.is_graph_captured and bs <= max(self.graph_bs):
            # if cuda graph captured
            pre_graph = self.pre_graphs[next(x for x in self.graph_bs if x >= bs)]
            post_graph = self.post_graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            graph_vars["hidden_states"][:bs] = hidden_states
            if residual is not None:
                graph_vars["residual"][:bs] = residual
            graph_vars["positions"][:bs] = position_ids

            pre_graph.replay()

            q = graph_vars["outputs_q"][:bs]
            k = graph_vars["outputs_k"][:bs]
            v = graph_vars["outputs_v"][:bs]

            attn_o = self.self_attn.attn(q, k, v, self.self_attn.sinks.data)

            graph_vars["attn_o"][:bs] = attn_o

            post_graph.replay()

            hidden_states = graph_vars["hidden_states"][:bs]
            residual = graph_vars["residual"][:bs]

            return hidden_states, residual
        else:
            if residual is None:
                residual = hidden_states.clone()
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
            )

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    def _pre_forward(
            self,
            positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None,
            slot_mapping: torch.Tensor | None,
            outputs_qkv: torch.Tensor | None = None,
            ):
        if residual is None:
            residual = hidden_states.clone()
            self.input_layernorm(hidden_states, out=hidden_states)
        else:
            self.input_layernorm(hidden_states, residual, out=hidden_states)

        qkv = self.self_attn.qkv(hidden_states, out=outputs_qkv)
        q, k, v = qkv.split([self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)

        self.self_attn.rotary_emb(positions, q, k)

        q = q.view(-1, self.self_attn.attn.num_heads, self.self_attn.attn.head_dim)
        k = k.view(-1, self.self_attn.attn.num_kv_heads, self.self_attn.attn.head_dim)
        v = v.view(-1, self.self_attn.attn.num_kv_heads, self.self_attn.attn.head_dim)

        store_kvcache(
            k,
            v,
            self.self_attn.attn.k_cache,
            self.self_attn.attn.v_cache,
            slot_mapping,
        )

        return residual, q, k, v

    def _post_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        hidden_states = self.self_attn.o_proj(hidden_states.view(-1, self.self_attn.num_heads * self.self_attn.head_dim))

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def capture_cudagraph(self, layer_idx: int = 0, max_num_batched_tokens: int = 1024, graph_vars: dict = None):
        self.is_graph_captured = True

        self.graph_vars = graph_vars
        hidden_states = graph_vars["hidden_states"]
        residual = graph_vars["residual"]
        positions = graph_vars["positions"]
        attn_o = graph_vars["attn_o"]
        outputs_q = graph_vars["outputs_q"]
        outputs_k = graph_vars["outputs_k"]
        outputs_v = graph_vars["outputs_v"]
        slot_mapping = graph_vars["slot_mapping"]

        self.graph_bs = list(range(1, 8, 1)) + list(range(8, 32, 4)) + list(range(32, 128, 8))
        bs = 128
        while bs <= min(max_num_batched_tokens, 1024):
            self.graph_bs.append(bs)
            bs = bs * 2
        self.pre_graphs = {}
        self.post_graphs = {}
        self.pre_graph_pool = None
        self.post_graph_pool = None

        for bs in reversed(self.graph_bs):
            pre_graph = torch.cuda.CUDAGraph()

            _positions = positions[:bs]
            _hidden_states = hidden_states[:bs]
            if layer_idx == 0:
                _residual = None
            else:
                _residual = residual[:bs]
            _slot_mapping = slot_mapping[:bs]

            _ = self._pre_forward(_positions, _hidden_states, _residual, _slot_mapping)

            with torch.cuda.graph(pre_graph, self.pre_graph_pool), capture():
                _positions = positions[:bs]
                _hidden_states = hidden_states[:bs]
                if layer_idx == 0:
                    _residual = None
                else:
                    _residual = residual[:bs]
                _slot_mapping = slot_mapping[:bs]

                _residual, _q, _k, _v = self._pre_forward(_positions, _hidden_states, _residual, _slot_mapping)

                outputs_q[:bs] = _q
                outputs_k[:bs] = _k
                outputs_v[:bs] = _v
                residual[:bs] = _residual

            post_graph = torch.cuda.CUDAGraph()

            _positions = positions[:bs]
            _attn_o = attn_o[:bs]
            _residual = residual[:bs]

            _ = self._post_forward(_attn_o, _residual)

            with torch.cuda.graph(post_graph, self.post_graph_pool), capture():
                _attn_o = attn_o[:bs]
                _residual = residual[:bs]

                _hidden_states, _residual = self._post_forward(_attn_o, _residual)

                hidden_states[:bs] = _hidden_states
                residual[:bs] = _residual

            if self.pre_graph_pool is None:
                self.pre_graph_pool = pre_graph.pool()
            if self.post_graph_pool is None:
                self.post_graph_pool = post_graph.pool()

            self.pre_graphs[bs] = pre_graph
            self.post_graphs[bs] = post_graph

            torch.cuda.synchronize()

        print(f"Captured {len(self.pre_graphs) + len(self.post_graphs)} CUDA graphs for layer {layer_idx}.")


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

        self.is_graph_captured = False
        self.num_stages = -1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = get_context()

        len_layered_prefill = context.len_prefill if context.is_prefill else 0

        hidden_states = self.embed_tokens(input_ids)

        if context.prefill_compute_layers:
            residual = hidden_states.clone()

            if intermediate_outputs is not None:
                i_hidden_states, i_residual = intermediate_outputs
                if i_hidden_states is not None:
                    hidden_states = torch.cat([i_hidden_states, hidden_states[i_hidden_states.size(0):]], dim=0)
                if i_residual is not None:
                    residual = torch.cat([i_residual, residual[i_residual.size(0):]], dim=0)

            pre_layers = tuple(np.arange(min(context.prefill_compute_layers)).tolist())
            post_layers = tuple(np.arange(max(context.prefill_compute_layers) + 1, len(self.layers)).tolist())

            bs = hidden_states[len_layered_prefill:].size(0)
            if bs > 0:
                if self.is_graph_captured and bs <= max(self.graph_bs):
                    self.graph_vars["hidden_states"][:bs] = hidden_states[len_layered_prefill:]
                    self.graph_vars["residual"][:bs] = residual[len_layered_prefill:]
                    self.graph_vars["positions"][:bs] = positions[len_layered_prefill:]
                    self.graph_vars["slot_mapping"].fill_(-1)
                    self.graph_vars["slot_mapping"][:bs] = context.slot_mapping[len_layered_prefill:]

                    is_prefill = context.is_prefill
                    len_prefill = context.len_prefill
                    context.is_prefill = False
                    context.len_prefill = 0

                    for layer_idx in pre_layers:
                        graph_idx = (layer_idx, next(x for x in self.graph_bs if x >= bs))
                        if layer_idx == 0:
                            pre_graph = self.pre_graphs[graph_idx]
                            pre_graph.replay()

                        qkv = self.graph_vars["outputs_qkv"][:bs]
                        q, k, v = qkv.split([self.layers[0].self_attn.q_size, self.layers[0].self_attn.kv_size, self.layers[0].self_attn.kv_size], dim=-1)

                        q = q.view(-1, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim)
                        k = k.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)
                        v = v.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)

                        self.layers[layer_idx].self_attn.attn.forward_attention(
                            self.graph_vars["attn_o"][:bs],
                            q, k, v,
                        )

                        if layer_idx < max(pre_layers):
                            con_graph = self.con_graphs[graph_idx]
                            con_graph.replay()
                        else:
                            post_graph = self.post_graphs[graph_idx]
                            post_graph.replay()

                    hidden_states[len_layered_prefill:] = self.graph_vars["hidden_states"][:bs]
                    residual[len_layered_prefill:] = self.graph_vars["residual"][:bs]

                    context.is_prefill = is_prefill
                    context.len_prefill = len_prefill
                else:
                    is_prefill = context.is_prefill
                    len_prefill = context.len_prefill
                    context.is_prefill = False
                    context.len_prefill = 0

                    slot_mapping = None
                    if context.slot_mapping is not None:
                        slot_mapping = context.slot_mapping.clone()
                        context.slot_mapping = slot_mapping[len_layered_prefill:]

                    _hidden_states = hidden_states[len_layered_prefill:]
                    _positions = positions[len_layered_prefill:]
                    _residual = residual[len_layered_prefill:]
                    for layer_idx in pre_layers:
                        if layer_idx == 0:
                            _hidden_states, _residual = self.layers[layer_idx](
                                _positions,
                                _hidden_states,
                                None,
                            )
                        else:
                            _hidden_states, _residual = self.layers[layer_idx](
                                _positions,
                                _hidden_states,
                                _residual
                            )
                    hidden_states[len_layered_prefill:] = _hidden_states
                    residual[len_layered_prefill:] = _residual

                    context.is_prefill = is_prefill
                    context.len_prefill = len_prefill
                    context.slot_mapping = slot_mapping


            bs = hidden_states.size(0)
            if self.is_graph_captured and bs <= max(self.graph_bs):
                self.graph_vars["hidden_states"][:bs] = hidden_states
                self.graph_vars["residual"][:bs] = residual
                self.graph_vars["positions"][:bs] = positions
                self.graph_vars["slot_mapping"].fill_(-1)
                self.graph_vars["slot_mapping"][:bs] = context.slot_mapping

                for layer_idx in sorted(context.prefill_compute_layers):
                    graph_idx = (layer_idx, next(x for x in self.graph_bs if x >= bs))
                    if layer_idx == min(context.prefill_compute_layers):
                        pre_graph = self.pre_graphs[graph_idx]
                        pre_graph.replay()

                    qkv = self.graph_vars["outputs_qkv"][:bs]
                    q, k, v = qkv.split([self.layers[0].self_attn.q_size, self.layers[0].self_attn.kv_size, self.layers[0].self_attn.kv_size], dim=-1)

                    q = q.view(-1, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim)
                    k = k.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)
                    v = v.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)

                    self.layers[layer_idx].self_attn.attn.forward_attention(
                        self.graph_vars["attn_o"][:bs],
                        q, k, v,
                    )

                    if layer_idx < max(context.prefill_compute_layers):
                        con_graph = self.con_graphs[graph_idx]
                        con_graph.replay()
                    else:
                        post_graph = self.post_graphs[graph_idx]
                        post_graph.replay()

                hidden_states[:] = self.graph_vars["hidden_states"][:bs]
                residual[:] = self.graph_vars["residual"][:bs]
            else:
                for layer_idx in sorted(context.prefill_compute_layers):
                    layer = self.layers[layer_idx]

                    if layer_idx == min(context.prefill_compute_layers):
                        _positions = positions[:]
                        _hidden_states = hidden_states[:]
                        _residual = residual[:] if layer_idx > 0 else None

                    _hidden_states, _residual = layer(_positions, _hidden_states, _residual)

                    if layer_idx == max(context.prefill_compute_layers):
                        hidden_states[:] = _hidden_states
                        residual[:] = _residual

            bs = hidden_states[len_layered_prefill:].size(0)
            if bs > 0:
                if self.is_graph_captured and bs <= max(self.graph_bs):
                    self.graph_vars["hidden_states"][:bs] = hidden_states[len_layered_prefill:]
                    self.graph_vars["positions"][:bs] = positions[len_layered_prefill:]
                    self.graph_vars["residual"][:bs] = residual[len_layered_prefill:]
                    self.graph_vars["slot_mapping"].fill_(-1)
                    self.graph_vars["slot_mapping"][:bs] = context.slot_mapping[len_layered_prefill:]

                    is_prefill = context.is_prefill
                    len_prefill = context.len_prefill
                    context.is_prefill = False
                    context.len_prefill = 0

                    for layer_idx in post_layers:
                        graph_idx = (layer_idx, next(x for x in self.graph_bs if x >= bs))
                        if layer_idx == min(post_layers):
                            pre_graph = self.pre_graphs[graph_idx]
                            pre_graph.replay()

                        qkv = self.graph_vars["outputs_qkv"][:bs]
                        q, k, v = qkv.split([self.layers[0].self_attn.q_size, self.layers[0].self_attn.kv_size, self.layers[0].self_attn.kv_size], dim=-1)

                        q = q.view(-1, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim)
                        k = k.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)
                        v = v.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)

                        self.layers[layer_idx].self_attn.attn.forward_attention(
                            self.graph_vars["attn_o"][:bs],
                            q, k, v,
                        )

                        if layer_idx < max(post_layers):
                            con_graph = self.con_graphs[graph_idx]
                            con_graph.replay()
                        else:
                            post_graph = self.post_graphs[graph_idx]
                            post_graph.replay()

                    hidden_states[len_layered_prefill:] = self.graph_vars["hidden_states"][:bs]
                    residual[len_layered_prefill:] = self.graph_vars["residual"][:bs]

                    context.is_prefill = is_prefill
                    context.len_prefill = len_prefill
                else:
                    is_prefill = context.is_prefill
                    len_prefill = context.len_prefill
                    context.is_prefill = False
                    context.len_prefill = 0

                    slot_mapping = None
                    if context.slot_mapping is not None:
                        slot_mapping = context.slot_mapping.clone()
                        context.slot_mapping = slot_mapping[len_layered_prefill:]

                    _hidden_states = hidden_states[len_layered_prefill:]
                    _positions = positions[len_layered_prefill:]
                    _residual = residual[len_layered_prefill:]
                    for layer_idx in post_layers:
                        if layer_idx == 0:
                            _hidden_states, _residual = self.layers[layer_idx](
                                _positions,
                                _hidden_states,
                                None,
                            )
                        else:
                            _hidden_states, _residual = self.layers[layer_idx](
                                _positions,
                                _hidden_states,
                                _residual,
                            )
                    hidden_states[len_layered_prefill:] = _hidden_states
                    residual[len_layered_prefill:] = _residual

                    context.is_prefill = is_prefill
                    context.len_prefill = len_prefill
                    context.slot_mapping = slot_mapping

            if len(self.layers) - 1 in context.prefill_compute_layers:
                _hidden_states, _residual = self.norm(hidden_states[:], residual[:])
                hidden_states[:] = _hidden_states
                residual[:] = _residual
            else:
                if hidden_states[len_layered_prefill:].size(0) > 0:
                    _hidden_states, _residual = self.norm(
                        hidden_states[len_layered_prefill:], residual[len_layered_prefill:]
                    )
                    hidden_states[len_layered_prefill:] = _hidden_states
                    residual[len_layered_prefill:] = _residual
        else:
            residual = None
            bs = hidden_states.size(0)
            if self.is_graph_captured and bs <= max(self.graph_bs):
                self.graph_vars["hidden_states"][:bs] = hidden_states
                self.graph_vars["positions"][:bs] = positions
                self.graph_vars["slot_mapping"].fill_(-1)
                self.graph_vars["slot_mapping"][:bs] = context.slot_mapping

                for layer_idx in range(len(self.layers)):
                    graph_idx = (layer_idx, next(x for x in self.graph_bs if x >= bs))
                    if layer_idx == 0:
                        pre_graph = self.pre_graphs[graph_idx]
                        pre_graph.replay()

                    qkv = self.graph_vars["outputs_qkv"][:bs]
                    q, k, v = qkv.split([self.layers[0].self_attn.q_size, self.layers[0].self_attn.kv_size, self.layers[0].self_attn.kv_size], dim=-1)

                    q = q.view(-1, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim)
                    k = k.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)
                    v = v.view(-1, self.layers[0].self_attn.num_kv_heads, self.layers[0].self_attn.head_dim)

                    self.layers[layer_idx].self_attn.attn.forward_attention(
                        self.graph_vars["attn_o"][:bs],
                        q, k, v,
                    )

                    if layer_idx < len(self.layers) - 1:
                        con_graph = self.con_graphs[graph_idx]
                        con_graph.replay()
                    else:
                        post_graph = self.post_graphs[graph_idx]
                        post_graph.replay()

                hidden_states = self.graph_vars["hidden_states"][:bs]
                residual = self.graph_vars["residual"][:bs]

                hidden_states, residual = self.norm(hidden_states, residual)
            else:
                for layer in self.layers:
                    hidden_states, residual = layer(positions, hidden_states, residual)

                hidden_states, residual = self.norm(hidden_states, residual)

        return hidden_states, residual

    def capture_cudagraph_layers(self, max_num_batched_tokens: int = 1024, schedule_mode: str = "chunked-prefill"):
        self.is_graph_captured = True

        hidden_size = self.embed_tokens.weight.size(1)

        hidden_states = torch.zeros(max_num_batched_tokens, hidden_size)
        residual = torch.zeros(max_num_batched_tokens, hidden_size)
        positions = torch.zeros(max_num_batched_tokens, dtype=torch.int64)
        attn_o = torch.zeros(max_num_batched_tokens, self.layers[0].self_attn.total_num_heads // dist.get_world_size(), self.layers[0].self_attn.head_dim)
        outputs_qkv = torch.zeros(
            max_num_batched_tokens,
            self.layers[0].self_attn.q_size + 2 * self.layers[0].self_attn.kv_size,
        )

        slot_mapping = torch.zeros(max_num_batched_tokens, dtype=torch.int32)

        self.graph_vars = dict(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            attn_o=attn_o,
            slot_mapping=slot_mapping,
            outputs_qkv=outputs_qkv,
        )

        self.graph_bs = list(range(1, 8, 1)) + list(range(8, 16, 2)) + list(range(16, 32, 4)) + list(range(32, 128, 8))
        bs = 128
        # self.graph_bs = []
        # bs = 1
        while bs <= min(max_num_batched_tokens, 1024):
            self.graph_bs.append(bs)
            bs = bs * 2
        self.graph_bs = [bs for bs in self.graph_bs if bs <= max_num_batched_tokens]
        self.pre_graphs = {}
        self.con_graphs = {}
        self.post_graphs = {}
        self.pre_graph_pool = None
        self.con_graph_pool = None
        self.post_graph_pool = None

        for layer_idx in range(len(self.layers)):
            for bs in reversed(self.graph_bs):
                if layer_idx == 0 or schedule_mode == "layered-prefill":
                    pre_graph = torch.cuda.CUDAGraph()

                    def func(positions, hidden_states, residual, slot_mapping, outputs_qkv, bs):
                        _positions = positions[:bs]
                        _hidden_states = hidden_states[:bs]
                        if layer_idx == 0:
                            _residual = None
                        else:
                            _residual = residual[:bs]
                        _slot_mapping = slot_mapping[:bs]
                        _outputs_qkv = outputs_qkv[:bs]

                        _residual, *_ = self.layers[layer_idx]._pre_forward(_positions, _hidden_states, _residual, _slot_mapping, outputs_qkv=_outputs_qkv)

                        if layer_idx == 0:
                            residual[:bs] = _residual

                    func(positions, hidden_states, residual, slot_mapping, outputs_qkv, bs)

                    with torch.cuda.graph(pre_graph, self.pre_graph_pool), capture():
                        func(positions, hidden_states, residual, slot_mapping, outputs_qkv, bs)

                    if self.pre_graph_pool is None:
                        self.pre_graph_pool = pre_graph.pool()
                    self.pre_graphs[(layer_idx, bs)] = pre_graph

                if layer_idx == len(self.layers) - 1 or schedule_mode == "layered-prefill":
                    post_graph = torch.cuda.CUDAGraph()

                    def func(attn_o, hidden_states, residual, bs):
                        _attn_o = attn_o[:bs]
                        _residual = residual[:bs]

                        _hidden_states, _residual = self.layers[layer_idx]._post_forward(_attn_o, _residual)
                        hidden_states[:bs] = _hidden_states
                        residual[:bs] = _residual

                    func(attn_o, hidden_states, residual, bs)

                    with torch.cuda.graph(post_graph, self.post_graph_pool), capture():
                        func(attn_o, hidden_states, residual, bs)

                    if self.post_graph_pool is None:
                        self.post_graph_pool = post_graph.pool()
                    self.post_graphs[(layer_idx, bs)] = post_graph

                if layer_idx < len(self.layers) - 1:
                    con_graph = torch.cuda.CUDAGraph()

                    def func(positions, hidden_states, residual, attn_o, slot_mapping, outputs_qkv, bs):
                        _positions = positions[:bs]
                        _hidden_states = hidden_states[:bs]
                        _residual = residual[:bs]
                        _attn_o = attn_o[:bs]
                        _slot_mapping = slot_mapping[:bs]
                        _outputs_qkv = outputs_qkv[:bs]

                        _hidden_states, _residual = self.layers[layer_idx]._post_forward(_attn_o, _residual)
                        _residual, *_ = self.layers[layer_idx + 1]._pre_forward(_positions, _hidden_states, _residual, _slot_mapping, outputs_qkv=_outputs_qkv)

                    func(positions, hidden_states, residual, attn_o, slot_mapping, outputs_qkv, bs)

                    with torch.cuda.graph(con_graph, self.con_graph_pool), capture():
                        func(positions, hidden_states, residual, attn_o, slot_mapping, outputs_qkv, bs)

                    if self.con_graph_pool is None:
                        self.con_graph_pool = con_graph.pool()

                    self.con_graphs[(layer_idx, bs)] = con_graph

                torch.cuda.synchronize()

            print(f"Captured {len(self.pre_graphs) + len(self.post_graphs) + len(self.con_graphs)} CUDA graphs for layer {layer_idx}.")


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv", "q"),
        "k_proj": ("qkv", "k"),
        "v_proj": ("qkv", "v"),
        # "gate_proj": ("gate_up_proj", 0),
        # "up_proj": ("gate_up_proj", 1),

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
        hidden_states = self.model(input_ids, positions, intermediate_outputs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

    def capture_cudagraph_layers(self, max_num_batched_tokens: int = 1024, schedule_mode: str = "chunked-prefill"):
        self.model.capture_cudagraph_layers(max_num_batched_tokens, schedule_mode)
