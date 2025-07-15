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
    for part in ["layers", str(layer_idx), "mlp", "experts"]:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        elif isinstance(layer, nn.ModuleList) and part.isdigit():
            layer = layer[int(part)]
        else:
            return None
    if isinstance(layer, FusedMoE):
        return layer
    return None

# MoE expert 파라미터 패턴: model.layers.<layer_idx>.mlp.experts.<expert_id>.<proj>.weight
moe_expert_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down_proj|up_proj|gate_proj)\.weight")

# proj -> (param_name, shard_id) 매핑
proj_to_param_shard = {
    "down_proj": ("w2_weight", "w2"),
    "gate_proj": ("w13_weight", "w1"),
    "up_proj":   ("w13_weight", "w3"),
}

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 1. MoE expert 파라미터 처리
                m = moe_expert_pattern.fullmatch(weight_name)
                if m:
                    layer_idx = int(m.group(1))
                    expert_id = int(m.group(2))
                    proj = m.group(3)
                    param_name, shard_id = proj_to_param_shard[proj]
                    # 실제 파라미터 이름: model.layers.<layer_idx>.mlp.experts.<param_name>
                    param_path = f"model.layers.{layer_idx}.mlp.experts.{param_name}"
                    moe_layer = find_fused_moe_layer(model, layer_idx)
                    if moe_layer is not None:
                        param = model.get_parameter(param_path)
                        moe_layer.weight_loader(param, f.get_tensor(weight_name), weight_name, shard_id, expert_id)
                    continue
                # 2. 기존 packed_modules_mapping 처리
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 3. 일반 파라미터
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except AttributeError as e:
                        print(f"[경고] 파라미터 {weight_name} 로딩 실패: {e}")
                        continue
