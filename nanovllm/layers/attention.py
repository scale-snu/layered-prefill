import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton 커널: Key-Value 캐시에 데이터를 저장하는 최적화된 커널

    Args:
        key_ptr: 입력 Key 텐서의 포인터
        key_stride: Key 텐서의 stride
        value_ptr: 입력 Value 텐서의 포인터
        value_stride: Value 텐서의 stride
        k_cache_ptr: Key 캐시 텐서의 포인터
        v_cache_ptr: Value 캐시 텐서의 포인터
        slot_mapping_ptr: 슬롯 매핑 텐서의 포인터
        D: 헤드 차원 (num_heads * head_dim)
    """
    idx = tl.program_id(0)  # 현재 스레드의 인덱스

    # Key와 Value의 오프셋 계산
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # Key와 Value 데이터 로드
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 슬롯 매핑에서 캐시 위치 가져오기
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)

    # 캐시에 Key와 Value 저장
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    Key-Value 캐시에 데이터를 저장하는 함수

    Args:
        key: 저장할 Key 텐서 [N, num_heads, head_dim]
        value: 저장할 Value 텐서 [N, num_heads, head_dim]
        k_cache: Key 캐시 텐서
        v_cache: Value 캐시 텐서
        slot_mapping: 슬롯 매핑 텐서 [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 메모리 레이아웃 검증 (성능 최적화를 위해)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # Triton 커널 실행
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    어텐션 레이어 클래스

    Flash Attention을 사용하여 효율적인 어텐션 계산을 수행합니다.
    PREFILLING과 DECODING 단계를 구분하여 처리하며,
    Staged-Prefill 모드에서는 단계별로 다른 레이어들을 계산할 수 있습니다.
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        window_size: int = -1,
    ):
        """
        어텐션 레이어 초기화

        Args:
            num_heads: 어텐션 헤드 수
            head_dim: 각 헤드의 차원
            scale: 어텐션 스케일링 팩터 (보통 1/sqrt(head_dim))
            num_kv_heads: Key-Value 헤드 수 (GQA/MQA 지원)
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size

        # KV 캐시 초기화 (빈 텐서로 시작)
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: None | torch.Tensor = None) -> torch.Tensor:
        """
        어텐션 계산 수행

        Args:
            q: Query 텐서 [total_tokens, num_heads * head_dim]
            k: Key 텐서 [total_tokens, num_kv_heads * head_dim]
            v: Value 텐서 [total_tokens, num_kv_heads * head_dim]

        Returns:
            어텐션 출력 [total_tokens, num_heads * head_dim]
        """
        o: torch.Tensor

        # 텐서를 헤드별로 재구성
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 현재 컨텍스트 정보 가져오기
        context = get_context()

        # PREFILLING 토큰 수 계산
        len_prefill = context.cu_seqlens_q[-1] if context.is_prefill else 0

        # KV 캐시 참조
        k_cache, v_cache = self.k_cache, self.v_cache

        # KV 캐시가 존재하면 현재 Key-Value를 캐시에 저장
        if k_cache.numel() and v_cache.numel():
            store_kvcache(
                k,
                v,
                k_cache,
                v_cache,
                context.slot_mapping,
            )

        os = []  # 출력 텐서들을 저장할 리스트

        # ===== PREFILLING 단계 처리 =====
        if context.is_prefill and len_prefill > 0:
            if context.prefill_block_tables is not None:
                # Chunked prefill 모드: 캐시된 Key-Value 사용
                k, v = k_cache, v_cache
            else:
                # 일반 prefill 모드: 현재 입력의 PREFILLING 부분만 사용
                k = k[:len_prefill]
                v = v[:len_prefill]

            # Flash Attention을 사용한 PREFILLING 어텐션 계산
            o = flash_attn_varlen_func(
                q[:len_prefill], k, v, learnable_sink=sinks,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.prefill_block_tables,
                window_size=(self.window_size, -1),
                # return_attn_probs=True,
            )
            # if sinks is not None:
            #     o_dtype = o.dtype
            #     lse = lse.transpose(-2, -1).unsqueeze(dim=-1)

            #     multiplier = 1 / (torch.exp(sinks.reshape(1, 1, -1, 1) - lse) + 1)
            #     o = (o * multiplier).to(o_dtype)
            o = o.view(-1, self.num_heads * self.head_dim)
            os.append(o)

        # ===== DECODING 단계 처리 =====
        if context.decode_block_tables is not None:  # decoding
            # Flash Attention with KV Cache를 사용한 DECODING 어텐션 계산
            o = flash_attn_with_kvcache(
                q[len_prefill:].unsqueeze(1),  # 마지막 토큰만 사용 (새로 생성된 토큰)
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.decode_block_tables,
                softmax_scale=self.scale,
                causal=True,
                # return_softmax_lse=True,
                learnable_sink=sinks,
                window_size=(self.window_size, -1),
            )
            # print(lse.flatten()[0])
            # if sinks is not None:
            #     o_dtype = o.dtype
            #     lse = lse.transpose(-2, -1).unsqueeze(dim=-1)

            #     multiplier = 1 / (torch.exp(sinks.reshape(1, 1, -1, 1) - lse) + 1)
            #     o = (o * multiplier).to(o_dtype)
            o = o.view(-1, self.num_heads * self.head_dim)
            os.append(o)

        # PREFILLING과 DECODING 출력을 연결
        o = torch.cat(os, dim=0)
        return o
