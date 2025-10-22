import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import re

import triton
import triton.language as tl

from nanovllm.layers.fused_moe import FusedMoE

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_all_fused_moe_layers(module):
    moes = []
    for child in module.children():
        if isinstance(child, FusedMoE):
            moes.append(child)
        else:
            moes.extend(get_all_fused_moe_layers(child))
    return moes


def find_fused_moe_layer(model, layer_idx):
    layer = model
    for part in ["model", "layers", str(layer_idx), "mlp", "experts"]:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        elif isinstance(layer, nn.ModuleList) and part.isdigit():
            layer = layer[int(part)]
        else:
            return None
    if isinstance(layer, FusedMoE):
        return layer
    return None

moe_expert_pattern = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts(?:\.(\d+))?\."
    r"(down_proj|up_proj|gate_proj|down_proj_bias|gate_up_proj_bias|gate_up_proj_blocks|down_proj_blocks|gate_up_proj_scales|down_proj_scales)(?:\.weight)?$"
)

proj_to_param_shard = {
    "down_proj": ("w2_weight", "w2"),
    "gate_proj": ("w13_weight", "w1"),
    "up_proj":   ("w13_weight", "w3"),
    "gate_up_proj_bias": ("w13_bias", "w13"),
    "down_proj_bias": ("w2_bias", "w2"),
    "gate_up_proj_blocks": ("w13_weight", "w13"),
    "down_proj_blocks": ("w2_weight", "w2"),
    "gate_up_proj_scales": ("w13_weight_scale", "w13"),
    "down_proj_scales": ("w2_weight_scale", "w2"),
}

def _dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                float_dtype: torch.dtype) -> torch.Tensor:
    from .mxfp import upcast_from_mxfp_torch

    return upcast_from_mxfp_torch(x, scale, float_dtype, axis=-1)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # print(f"Loading weight: {weight_name}")
                m = moe_expert_pattern.fullmatch(weight_name)
                if m:
                    layer_idx = int(m.group(1))
                    expert_id = int(m.group(2)) if m.group(2) else 0
                    proj = m.group(3)
                    param_name, shard_id = proj_to_param_shard[proj]

                    param_path = f"model.layers.{layer_idx}.mlp.experts.{param_name}"
                    moe_layer = find_fused_moe_layer(model, layer_idx)
                    assert moe_layer is not None, f"FusedMoE layer not found for layer index {layer_idx}"
                    loaded_weight = f.get_tensor(weight_name)
                    if moe_layer.params_dtype in [torch.float16, torch.bfloat16, "float16", "bfloat16"]:
                        if weight_name.endswith("_scales"):
                            continue
                        param = model.get_parameter(param_path)
                        scale = None
                        if proj in ["gate_up_proj_blocks", "down_proj_blocks"]:
                            scale_name = f"model.layers.{layer_idx}.mlp.experts.{proj.replace('_blocks', '_scales')}"
                            scale = f.get_tensor(scale_name)
                        if scale is not None:
                            loaded_weight = loaded_weight.cuda()
                            scale = scale.cuda()
                            if scale.ndim + 1 == loaded_weight.ndim:
                                scale = scale.unsqueeze(-1)
                            loaded_weight = _dequant_mxfp4(loaded_weight, scale, param.dtype)

                        moe_layer.weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)
                    else:
                        param = model.get_parameter(param_path)
                        assert moe_layer.params_dtype == "mxfp4", "Only mxfp4 quantization is supported for MoE layers."

                        def cdiv(a, b):
                            return -(-a // b)

                        mxfp4_block = 32
                        num_experts = moe_layer.local_num_experts

                        tp_rank = moe_layer.tp_rank
                        tp_size = moe_layer.tp_size

                        intermediate_size = moe_layer.intermediate_size
                        intermediate_size_block = intermediate_size // mxfp4_block
                        per_rank_intermediate_size_block = cdiv(intermediate_size_block,
                                                                tp_size)
                        per_rank_intermediate_size = (per_rank_intermediate_size_block *
                                                    mxfp4_block)

                        tp_rank_start = tp_rank * per_rank_intermediate_size
                        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size,
                                        intermediate_size)

                        if param_name == "w13_weight_scale":
                            narrow_weight = loaded_weight[:, 2 * tp_rank_start:2 * tp_rank_end]
                            moe_layer.weight_loader(
                                param,
                                narrow_weight,
                                weight_name,
                                shard_id="all",
                                expert_id=None,
                            )
                        elif param_name == "w2_weight_scale":
                            narrow_weight = loaded_weight[..., tp_rank_start // mxfp4_block:tp_rank_end // mxfp4_block]
                            print(narrow_weight.shape)
                            print(param.data.shape)
                            moe_layer.weight_loader(
                                param,
                                narrow_weight,
                                weight_name,
                                shard_id="all",
                                expert_id=None,
                            )
                        elif param_name == "w13_weight":
                            loaded_weight = loaded_weight.view(num_experts, 2 * intermediate_size, -1).contiguous()
                            narrow_weight = loaded_weight[:, 2 * tp_rank_start:2 * tp_rank_end]
                            moe_layer.weight_loader(
                                param,
                                narrow_weight,
                                weight_name,
                                shard_id="all",
                                expert_id=None,
                            )
                        elif param_name == "w2_weight":
                            loaded_weight = loaded_weight.view(num_experts, -1, intermediate_size // 2).contiguous()
                            narrow_weight = loaded_weight[..., tp_rank_start // 2:tp_rank_end // 2]
                            moe_layer.weight_loader(
                                param,
                                narrow_weight,
                                weight_name,
                                shard_id="all",
                                expert_id=None,
                            )
                        elif param_name == "w13_bias":
                            narrow_weight = loaded_weight[:, 2 * tp_rank_start:2 * tp_rank_end]
                            moe_layer.weight_loader(
                                param,
                                narrow_weight,
                                weight_name,
                                shard_id="all",
                                expert_id=None,
                            )
                        elif param_name == "w2_bias":
                            if tp_rank == 0:
                                moe_layer.weight_loader(
                                    param,
                                    loaded_weight,
                                    weight_name,
                                    shard_id="all",
                                    expert_id=None,
                                )
                        else:
                            raise ValueError(f"Unknown param_name {param_name} for MoE layer.")
                    continue
                if "sinks" in weight_name:
                    import torch.distributed as dist

                    tp_rank = dist.get_rank()
                    tp_size = dist.get_world_size()

                    param = model.get_parameter(weight_name)
                    param.data.copy_(f.get_tensor(weight_name).chunk(tp_size, 0)[tp_rank])
                    continue

                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except AttributeError as e:
                        print(f"[Warning] Parameter {weight_name} not found in the model.")
                        continue

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)

        if quant_method is not None:
            if hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(module)
