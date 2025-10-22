# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional, Union, overload

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import UninitializedParameter
# from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

from nanovllm.layers.custom_all_reduce import tensor_model_parallel_all_reduce
# from nanovllm.layers.nccl_communicator import tensor_model_parallel_all_reduce
from nanovllm.layers.for_moe.utils import set_weight_attrs
from nanovllm.layers.for_moe.fused_moe import fused_experts
from nanovllm.layers.for_moe.fused_moe_fp4 import fused_marlin_moe
from nanovllm.utils.scalar_type import scalar_types
from nanovllm import ops
from nanovllm.utils.utils import ForkedPdb

fused_moe_pallas = None  # type: ignore


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def nvfp4_marlin_process_scales(marlin_scales):
    if not (marlin_scales >= 0).all():
        print(
            "NVFP4 Marlin assumes the scales to be >=0, but has encountered "
            "negative scales. Accuracy will likely be degraded. This is "
            "because it changes the scales from FP8-S1E4M3 to a special "
            "FP8-S0E5M3 format to speedup the dequantization.")

    # convert to half first, we would convert to fp8 later
    marlin_scales = marlin_scales.to(torch.half)

    # 8 is the number of scale number using by one thread
    marlin_scales = marlin_scales.view(marlin_scales.size(0) // 2, 2, -1, 8)
    marlin_scales = marlin_scales.permute(0, 2, 1, 3).reshape(
        marlin_scales.size(0) * 2, -1)

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1)

    # We assume that weight_scale (FP8-S1E4M3) is always greater
    # than or equal to 0. So we can convert
    # (weight_scale * (2 ** 7) to a special FP8-S0E5M3 format.
    # After multiplying by 2 ** 7, the top bit of FP8-S0E5M3 would always be 1
    # when weight_scale > 0. This allows us to have an exponent bias
    # closer to zero after dequantization.

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()

    return marlin_scales


def mxfp4_marlin_process_scales(marlin_scales):
    # 8 is the number of scale number using by one thread
    marlin_scales = marlin_scales.view(marlin_scales.size(0) // 2, 2, -1, 8)
    marlin_scales = marlin_scales.permute(0, 2, 1, 3).reshape(
        marlin_scales.size(0) * 2, -1)

    # fit the layout of fp8 dequantization
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1)
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    return marlin_scales


def nvfp4_marlin_process_global_scale(global_scale):
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    if global_scale.dtype == torch.half:
        target_exponent = 5
    elif global_scale.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 1 = 14
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 1 = 126
    exponent_bias = 2**(target_exponent - 1) - 2**(fp4_exponent - 1)
    return global_scale * (2.0**(exponent_bias - 7))


def prepare_moe_fp4_layer_for_marlin(layer: torch.nn.Module) -> None:
    print(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads.")

    is_nvfp4 = hasattr(layer, "w13_weight_scale_2")
    group_size = 16 if is_nvfp4 else 32

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = torch.get_default_dtype()
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k // 2)

        for i in range(e):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(qweight,
                                                    perm,
                                                    size_k,
                                                    size_n,
                                                    4)
            tensor_list.append(marlin_qweight)

        weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        weight = torch.nn.Parameter(weight, requires_grad=False)

        setattr(layer, name, weight)

    # WEIGHT SCALES
    # Permute scales
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale")
        if not is_nvfp4:
            scales = scales.view(torch.float8_e8m0fnu)
        scales = scales.to(param_dtype)
        if is_nvfp4:
            global_scale = getattr(layer,
                                   name + "_weight_scale_2").to(param_dtype)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        for i in range(e):
            scale = scales[i].T

            marlin_scales = marlin_permute_scales(s=scale,
                                                  size_k=size_k,
                                                  size_n=size_n,
                                                  group_size=group_size)
            if is_nvfp4:
                marlin_scales = nvfp4_marlin_process_scales(marlin_scales)
            else:
                marlin_scales = mxfp4_marlin_process_scales(marlin_scales)
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        scales = torch.nn.Parameter(scales, requires_grad=False)
        setattr(layer, name + "_weight_scale", scales)

        if is_nvfp4:
            global_scale = nvfp4_marlin_process_global_scale(global_scale)
            global_scale = torch.nn.Parameter(global_scale,
                                              requires_grad=False)
            setattr(layer, name + "_weight_scale_2", global_scale)

    # BIAS
    # Permute bias
    for name in ["w13_bias", "w2_bias"]:
        if not hasattr(layer, name):
            continue
        bias = getattr(layer, name).to(param_dtype)

        tensor_list = []
        for i in range(e):
            expert_bias = bias[i]

            tensor_list.append(marlin_permute_bias(expert_bias))

        bias = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        bias = torch.nn.Parameter(bias, requires_grad=False)
        setattr(layer, name, bias)


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    tp_rank: int

    @staticmethod
    def make():
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()
        return FusedMoEParallelConfig(tp_size=tp_size, tp_rank=tp_rank)

@dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    in_dtype: torch.dtype  # The activation type.
    quant_dtype: torch.dtype = None

    # TODO: add more quantization params, blocked, per-token, etc.
    block_size: int = 128

    max_num_tokens: int = 256

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(ABC):

    moe: MoEConfig

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def init_prepare_finalize(self, moe: MoEConfig):
        self.moe = moe
        self.topk_indices_dtype = None

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

# count_tensor: torch.Tensor
# count: torch.Tensor

class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MoE method with mxfp4 quantization."""

    def __init__(self, moe: MoEConfig):
        super().__init__()
        self.topk_indices_dtype = None
        self.moe = moe
        self.max_capture_size = moe.max_num_tokens

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype,
                       has_bias: bool = False,
                       **extra_weight_attrs):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition
        intermediate_size_per_partition_after_pad = (
            (intermediate_size_per_partition_after_pad + 128 - 1)
            // 128 * 128
        )
        hidden_size = (hidden_size + 256 - 1) // 256 * 256

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = \
            intermediate_size_per_partition_after_pad

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        prepare_moe_fp4_layer_for_marlin(layer)

    #     layer.gemm1_alpha = torch.nn.Parameter(torch.tensor(
    #         [1.702] * self.num_experts, dtype=torch.float32).cuda(),
    #                                   requires_grad=False)
    #     layer.gemm1_beta = torch.nn.Parameter(torch.tensor(
    #         [1.0] * self.num_experts, dtype=torch.float32).cuda(),
    #                                  requires_grad=False)
    #     layer.gemm1_clamp_limit = torch.nn.Parameter(torch.tensor(
    #         [7.0] * self.num_experts, dtype=torch.float32).cuda(),
    #                                         requires_grad=False)

    #     sf_block_size = 32  # mxfp4 block size

    #     # Common shape assertions
    #     assert (layer.w13_weight.dim() == 3
    #             and layer.w13_weight.shape[0] == self.num_experts
    #             and layer.w13_weight.shape[1] == self.intermediate_size * 2
    #             and layer.w13_weight.shape[2] == self.hidden_size // 2)
    #     assert (layer.w13_weight_scale.dim() == 3
    #             and layer.w13_weight_scale.shape[0] == self.num_experts
    #             and layer.w13_weight_scale.shape[1]
    #             == self.intermediate_size * 2
    #             and layer.w13_weight_scale.shape[2]
    #             == self.hidden_size // sf_block_size), \
    #         f"{layer.w13_weight_scale.shape} vs {self.num_experts}, {self.intermediate_size * 2}, {self.hidden_size // sf_block_size}"
    #     assert (layer.w2_weight.dim() == 3
    #             and layer.w2_weight.shape[0] == self.num_experts
    #             and layer.w2_weight.shape[1] == self.hidden_size and
    #             layer.w2_weight.shape[2] == self.intermediate_size // 2)
    #     assert (layer.w2_weight_scale.dim() == 3
    #             and layer.w2_weight_scale.shape[1] == self.hidden_size
    #             and layer.w2_weight_scale.shape[2]
    #             == self.intermediate_size // sf_block_size)
    #     assert (layer.w13_bias.dim() == 2
    #             and layer.w13_bias.shape[0] == self.num_experts
    #             and layer.w13_bias.shape[1] == self.intermediate_size * 2)
    #     assert (layer.w2_bias.dim() == 2
    #             and layer.w2_bias.shape[0] == self.num_experts
    #             and layer.w2_bias.shape[1] == self.hidden_size)

    #     # De-interleave and swap for w13 weight, bias, and scales
    #     w13_w = layer.w13_weight.data
    #     gate_w, up_w = w13_w[:, ::2, :], w13_w[:, 1::2, :]
    #     deinterleaved_w13_w = torch.cat([gate_w, up_w], dim=1)
    #     w1_w, w3_w = torch.chunk(deinterleaved_w13_w, 2, dim=1)
    #     w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)

    #     w13_b = layer.w13_bias.data.to(torch.float32)
    #     gate_b, up_b = w13_b[:, ::2], w13_b[:, 1::2]
    #     deinterleaved_w13_b = torch.cat([gate_b, up_b], dim=1)
    #     b1, b3 = torch.chunk(deinterleaved_w13_b, 2, dim=-1)
    #     w13_bias_swapped = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

    #     w13_s = layer.w13_weight_scale.data
    #     gate_s, up_s = w13_s[:, ::2, :], w13_s[:, 1::2, :]
    #     deinterleaved_w13_s = torch.cat([gate_s, up_s], dim=1)
    #     s1, s3 = torch.chunk(deinterleaved_w13_s, 2, dim=1)
    #     w13_scale_swapped = torch.cat([s3, s1], dim=1)

    #     def _interleave_mxfp4_cutlass_sm90(w):
    #         w_shape = w.shape
    #         w_interleaved = w.reshape(w_shape[0], w_shape[1],
    #                                     (w_shape[2] // 4), 4)
    #         w_interleaved = w_interleaved.permute(0, 2, 1, 3)
    #         w_interleaved = w_interleaved.reshape(
    #             w_shape[0], w_shape[2] // 4, w_shape[1] * 4)
    #         return w_interleaved

    #     w31_scales = w13_scale_swapped.to(torch.uint8).view(
    #         torch.uint8)
    #     w31_scales_interleaved = _interleave_mxfp4_cutlass_sm90(
    #         w31_scales)

    #     w2_weight_scale = layer.w2_weight_scale.data
    #     w2_scales = w2_weight_scale.to(torch.uint8).view(torch.uint8)
    #     w2_scales_interleaved = _interleave_mxfp4_cutlass_sm90(
    #         w2_scales)

    #     layer.w13_weight = torch.nn.Parameter(torch.cat([w3_w, w1_w],
    #                                                     dim=1),
    #                                             requires_grad=False)
    #     layer.w13_bias = torch.nn.Parameter(w13_bias_swapped,
    #                                         requires_grad=False)
    #     layer.w13_weight_scale = torch.nn.Parameter(
    #         w31_scales_interleaved, requires_grad=False)
    #     layer.w2_weight_scale = torch.nn.Parameter(
    #         w2_scales_interleaved, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        og_hidden_size = x.size(1)
        if og_hidden_size < self.hidden_size:
            pad_size = self.hidden_size - og_hidden_size
            x = F.pad(x, (0, pad_size), "constant", 0.0)

        return fused_marlin_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_bias,
            layer.w2_bias,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            global_scale1=None,
            global_scale2=None,
            quant_type_id=scalar_types.float4_e2m1f.id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            activation=activation,
            expert_map=expert_map)[..., :og_hidden_size].contiguous()

        assert x.dtype == torch.bfloat16

        # quant_scales = [
        #     layer.w13_weight_scale,
        #     layer.w2_weight_scale,
        # ]

        # fi_input = x
        # extra_kwargs = dict(
        #     use_w4_group_scaling=True,
        #     fc1_expert_weights=layer.w13_weight,
        #     fc2_expert_weights=layer.w2_weight,
        # )

        # output = torch.empty(x.size(0), self.hidden_size, dtype=torch.bfloat16, device=x.device)

        # _ = flashinfer_cutlass_fused_moe(
        #     input=fi_input,
        #     token_selected_experts=topk_ids.to(torch.int).contiguous(),
        #     token_final_scales=topk_weights,
        #     output_dtype=torch.bfloat16,
        #     output=output,
        #     quant_scales=quant_scales,
        #     fc1_expert_biases=layer.w13_bias,
        #     fc2_expert_biases=layer.w2_bias,
        #     swiglu_alpha=layer.gemm1_alpha,
        #     swiglu_beta=layer.gemm1_beta,
        #     swiglu_limit=layer.gemm1_clamp_limit,
        #     tp_size=self.moe.tp_size,
        #     tp_rank=self.moe.tp_rank,
        #     tune_max_num_tokens=self.max_capture_size,
        #     **extra_kwargs,
        # )

        print(output)

        return output[:, :x.size(1)]


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def __init__(self, moe: MoEConfig):
        super().__init__()
        self.fused_experts = fused_experts  # type: ignore
        self.topk_indices_dtype = None
        self.moe = moe

        self.rocm_aiter_fused_experts = None  # type: ignore

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype,
                       has_bias: bool = False,
                       **extra_weight_attrs):
        params_dtype = torch.bfloat16 if params_dtype == "bfloat16" else params_dtype
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if has_bias:
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # global count_tensor, count
        # count_tensor = torch.zeros(num_experts, dtype=torch.int32).cuda()
        # count = torch.zeros(num_experts, dtype=torch.int32).cuda()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        # global count_tensor, count
        # count.zero_()
        # count[topk_ids.flatten()] += 1
        # count_tensor.add_((count > 0).int())

        return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_bias=layer.w13_bias if hasattr(layer, "w13_bias") else None,
                w2_bias=layer.w2_bias if hasattr(layer, "w2_bias") else None,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

class FusedMoE(torch.nn.Module):

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        num_redundant_experts: int = 0,
        has_bias=False,
        max_num_tokens: int = 256,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make())

        self.global_num_experts = num_experts + num_redundant_experts

        self.layer_name = prefix

        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None

        # Determine expert maps
        self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        self.top_k = top_k

        self.intermediate_size = intermediate_size
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        # Only support float8 for now.
        quant_dtype = params_dtype

        moe = MoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=params_dtype,
            quant_dtype=quant_dtype,
            max_num_tokens=max_num_tokens,
        )
        self.moe_config = moe

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        if params_dtype == "mxfp4":
            quant_method = Mxfp4MoEMethod(moe)
        else:
            quant_method = UnquantizedFusedMoEMethod(moe)

        self.quant_method = quant_method

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "has_bias": has_bias,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if expert_data.ndim != loaded_weight.ndim:
            loaded_weight = loaded_weight.reshape(*loaded_weight.shape[:-2], -1)

        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)
        elif shard_id in ("w13",):
            # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
            shard_size = expert_data.shape[1]
            loaded_w = loaded_weight.narrow(1, shard_size * tp_rank, shard_size)
            expert_data.narrow(1, 0, shard_size).copy_(loaded_w)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int,
                  shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[False]) -> None:
        ...

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[True]) -> bool:
        ...

    def weight_loader(self,
                      param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      weight_name: str,
                      shard_id: str,
                      expert_id: int,
                      return_success: bool = False) -> Optional[bool]:
        if shard_id == "all":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        quant_method_name = self.quant_method.__class__.__name__

        if shard_id not in ("w1", "w2", "w3", "all", "w13"):
            raise ValueError(f"shard_id must be ['w1','w2','w3', 'all'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0, "w13": 1}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3 or "bias" in weight_name or "block" in weight_name
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if ("compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            if ('weight_scale_2' in weight_name
                    or 'input_scale' in weight_name):
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return True if return_success else None

        # Case model weights
        if "block" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=-1,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return True if return_success else None

        if "bias" in weight_name:
            if shard_id == "w2":
                if self.tp_rank == 0:
                    expert_data.copy_(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(-1, self.tp_rank * loaded_weight.size(-1) // self.tp_size, loaded_weight.size(-1) // self.tp_size)
                expert_data.copy_(loaded_weight)
            return True if return_success else None

        raise ValueError(f"Unrecognized weight_name {weight_name}.")

        return False if return_success else None

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        assert all(weight.is_contiguous() for _, weight in weights)

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1) for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        indices_type: Optional[torch.dtype] = None,
        expert_map: Optional[torch.Tensor] = None,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids) (tuple[torch.Tensor, torch.Tensor]):
            The weights and *global physical* expert ids of the top-k experts.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from nanovllm.layers.for_moe.fused_moe import fused_topk

        if custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        return False

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None
        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )

        # if hidden_states.shape[0] <= 10:
        #     print(self.tp_rank, final_hidden_states)

        if self.reduce_results and (self.tp_size > 1):
            # Default set to False. (May have to add shared expert outputs.
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        # if hidden_states.shape[0] <= 10:
        #     print(self.tp_rank, final_hidden_states)
        #     exit(0)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:

        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        global_physical_to_logical_map = list(range(num_experts))
        global_physical_to_logical_map += [
            i % num_experts for i in range(num_redundant_experts)
        ]
        physical_to_logical_map = global_physical_to_logical_map

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s