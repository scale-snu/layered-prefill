import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


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
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = get_context()

        len_staged_prefill = context.cu_seqlens_q[-1] if context.is_prefill else 0

        hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states.clone()
        if context.prefill_compute_layers:
            if intermediate_outputs is not None:
                i_hidden_states, i_residual = intermediate_outputs
                if i_hidden_states is not None:
                    hidden_states = torch.cat([i_hidden_states, hidden_states[i_hidden_states.size(0):]], dim=0)
                if i_residual is not None:
                    residual = torch.cat([i_residual, residual[i_residual.size(0):]], dim=0)

            prefill_compute_layer_map = {}
            for idx, prefill_compute_layer in enumerate(context.prefill_compute_layers):
                for layer_idx in prefill_compute_layer:
                    prefill_compute_layer_map[layer_idx] = idx

            for layer_idx, layer in enumerate(self.layers):
                if layer_idx in prefill_compute_layer_map:
                    idx = prefill_compute_layer_map[layer_idx]
                    start_idx = context.cu_seqlens_q[idx]
                    end_idx = context.cu_seqlens_q[idx + 1]
                    indices = torch.cat([
                        torch.arange(start_idx, end_idx, device=hidden_states.device),
                        torch.arange(len_staged_prefill, hidden_states.size(0), device=hidden_states.device)
                    ])
                    max_seqlen_k = context.max_seqlen_k
                    max_seqlen_q = context.max_seqlen_q
                    context.max_seqlen_k = int(context.cu_seqlens_k[idx + 1] - context.cu_seqlens_k[idx])
                    context.max_seqlen_q = int(context.cu_seqlens_q[idx + 1] - context.cu_seqlens_q[idx])
                    cu_seqlens_k = context.cu_seqlens_k.clone()
                    cu_seqlens_q = context.cu_seqlens_q.clone()
                    context.cu_seqlens_k = cu_seqlens_k[idx:idx + 2] - cu_seqlens_k[idx]
                    context.cu_seqlens_q = cu_seqlens_q[idx:idx + 2] - cu_seqlens_q[idx]
                    prefill_block_tables = None
                    if context.prefill_block_tables is not None:
                        prefill_block_tables = context.prefill_block_tables.clone()
                        context.prefill_block_tables = prefill_block_tables[start_idx:end_idx]
                    slot_mapping = None
                    if context.slot_mapping is not None:
                        slot_mapping = context.slot_mapping.clone()
                        context.slot_mapping = context.slot_mapping[indices]

                    _hidden_states, _residual = layer(positions[indices], hidden_states[indices], residual[indices])
                    hidden_states[indices] = _hidden_states
                    residual[indices] = _residual
                    context.max_seqlen_k = max_seqlen_k
                    context.max_seqlen_q = max_seqlen_q
                    context.cu_seqlens_k = cu_seqlens_k
                    context.cu_seqlens_q = cu_seqlens_q
                    context.prefill_block_tables = prefill_block_tables
                    context.slot_mapping = slot_mapping
                else:
                    if hidden_states[len_staged_prefill:].size(0) > 0:
                        is_prefill = context.is_prefill
                        context.is_prefill = False
                        slot_mapping = None
                        if context.slot_mapping is not None:
                            slot_mapping = context.slot_mapping.clone()
                            context.slot_mapping = slot_mapping[len_staged_prefill:]
                        _hidden_states, _residual = layer(
                            positions[len_staged_prefill:],
                            hidden_states[len_staged_prefill:],
                            residual[len_staged_prefill:]
                        )
                        hidden_states[len_staged_prefill:] = _hidden_states
                        residual[len_staged_prefill:] = _residual
                        context.is_prefill = is_prefill
                        context.slot_mapping = slot_mapping

            if len(self.layers) - 1 in prefill_compute_layer_map:
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
                if hidden_states[len_staged_prefill:].size(0) > 0:
                    _hidden_states, _residual = self.norm(
                        hidden_states[len_staged_prefill:], residual[len_staged_prefill:]
                    )
                    hidden_states[len_staged_prefill:] = _hidden_states
                    residual[len_staged_prefill:] = _residual
        else:
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)

            hidden_states, residual = self.norm(hidden_states, residual)

        return hidden_states, residual


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
