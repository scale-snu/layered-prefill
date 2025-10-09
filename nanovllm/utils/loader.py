import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import re

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
    r"(down_proj|up_proj|gate_proj|down_proj_bias|gate_up_proj_bias|gate_up_proj_blocks|down_proj_blocks)(?:\.weight)?$"
)

proj_to_param_shard = {
    "down_proj": ("w2_weight", "w2"),
    "gate_proj": ("w13_weight", "w1"),
    "up_proj":   ("w13_weight", "w3"),
    "gate_up_proj_bias": ("w13_bias", "w13"),
    "down_proj_bias": ("w2_bias", "w2"),
    "gate_up_proj_blocks": ("w13_weight", "w13"),
    "down_proj_blocks": ("w2_weight", "w2"),
}

def _dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                        "MX-FP4 models. Please install it with `pip install "
                        "amd-quark`.") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                m = moe_expert_pattern.fullmatch(weight_name)
                if m:
                    layer_idx = int(m.group(1))
                    expert_id = int(m.group(2)) if m.group(2) else 0
                    proj = m.group(3)
                    param_name, shard_id = proj_to_param_shard[proj]
                    scale = None
                    if proj in ["gate_up_proj_blocks", "down_proj_blocks"]:
                        scale_name = f"model.layers.{layer_idx}.mlp.experts.{proj.replace('_blocks', '_scales')}"
                        scale = f.get_tensor(scale_name)

                    param_path = f"model.layers.{layer_idx}.mlp.experts.{param_name}"
                    moe_layer = find_fused_moe_layer(model, layer_idx)
                    assert moe_layer is not None, f"FusedMoE layer not found for layer index {layer_idx}"
                    param = model.get_parameter(param_path)
                    loaded_weight = f.get_tensor(weight_name)
                    if scale is not None:
                        loaded_weight = loaded_weight.cuda()
                        scale = scale.cuda()
                        if scale.ndim + 1 == loaded_weight.ndim:
                            scale = scale.unsqueeze(-1)
                        loaded_weight = _dequant_mxfp4(loaded_weight, scale, param.dtype)

                    moe_layer.weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)
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
