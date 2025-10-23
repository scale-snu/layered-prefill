import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
import numpy as np
from collections import defaultdict

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.custom_all_reduce import capture
from nanovllm.utils.context import get_context, set_context, reset_context


class Qwen3Attention(nn.Module):
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
        super().__init__()
        tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)

        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()

        self.is_graph_captured = False

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

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

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

        qkv = self.self_attn.qkv_proj(hidden_states, out=outputs_qkv)
        q, k, v = qkv.split([self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)

        q_by_head = q.view(-1, self.self_attn.num_heads, self.self_attn.head_dim)
        self.self_attn.q_norm(q_by_head, out=q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(-1, self.self_attn.num_kv_heads, self.self_attn.head_dim)
        self.self_attn.k_norm(k_by_head, out=k_by_head)
        k = k_by_head.view(k.shape)

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


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

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

        self.graph_bs = list(range(1, 8, 1)) + list(range(8, 16, 2)) + list(range(16, 32, 4)) + list(range(32, 128, 8)) + list(range(128, 512, 64))
        bs = 512
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


class Qwen3ForCausalLM(nn.Module):
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
        super().__init__()
        self.model = Qwen3Model(config)

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

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
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    def capture_cudagraph_layers(self, max_num_batched_tokens: int = 1024, schedule_mode: str = "chunked-prefill"):
        self.model.capture_cudagraph_layers(max_num_batched_tokens, schedule_mode)
