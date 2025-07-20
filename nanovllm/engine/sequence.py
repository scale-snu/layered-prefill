from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    시퀀스의 처리 상태를 나타내는 열거형
    
    - WAITING: 대기 중 (스케줄러의 waiting 큐에 있음)
    - PREFILLING: 프롬프트 처리 중 (프롬프트 토큰들을 처리하는 단계)
    - DECODING: 디코딩 중 (새로운 토큰을 생성하는 단계)
    - FINISHED: 완료됨 (EOS 토큰 생성 또는 최대 토큰 수 도달)
    """
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()


class Sequence:
    """
    텍스트 생성 시퀀스를 관리하는 클래스
    
    하나의 요청(프롬프트)에 대한 전체 생명주기를 관리하며,
    토큰 처리, 메모리 블록 할당, 상태 관리 등을 담당합니다.
    
    Staged-Prefill 모드에서는 단계별 처리 정보도 함께 관리합니다.
    """
    block_size = 256  # KV 캐시 블록 크기 (고정값)
    counter = count()  # 시퀀스 ID 생성을 위한 카운터

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), seq_id: str = None):
        """
        시퀀스 초기화
        
        Args:
            token_ids: 프롬프트 토큰 ID 리스트
            sampling_params: 샘플링 파라미터 (온도, 최대 토큰 수 등)
            seq_id: 시퀀스 ID (None이면 자동 생성)
        """
        if seq_id is None:
            seq_id = next(Sequence.counter)
        self.seq_id = seq_id
        
        # 기본 상태 및 토큰 정보
        self.status = SequenceStatus.WAITING  # 초기 상태는 대기
        self.token_ids = copy(token_ids)      # 토큰 ID 리스트 복사
        self.last_token = token_ids[-1]       # 마지막 토큰 (디코딩에서 사용)
        self.num_tokens = len(self.token_ids) # 전체 토큰 수
        self.num_prompt_tokens = len(token_ids) # 프롬프트 토큰 수 (초기에는 전체와 동일)
        
        # 처리 진행 상황 추적
        self.num_processed_tokens = 0    # 지금까지 처리된 토큰 수
        self.num_tokens_to_process = 0   # 이번 스텝에서 처리할 토큰 수
        
        # 메모리 블록 관리
        self.block_table = []  # KV 캐시에서 할당된 블록 번호들
        
        # 샘플링 파라미터
        self.temperature = sampling_params.temperature  # 온도 (높을수록 더 랜덤)
        self.max_tokens = sampling_params.max_tokens    # 최대 생성 토큰 수
        self.ignore_eos = sampling_params.ignore_eos    # EOS 토큰 무시 여부
        
        # 출력 추적
        self.num_generated_from_last = self.num_tokens  # 마지막 출력 이후 생성된 토큰 수
        
        # ===== STAGED-PREFILL 관련 필드들 =====
        # stage: 현재 시퀀스가 처리 중인 단계
        #   -1: 새로운 시퀀스 (아직 단계에 진입하지 않음)
        #   0~num_stages-1: 각 단계별 처리 중
        #   num_stages: 모든 단계 완료 (디코딩 단계로 전환)
        self.stage = -1
        
        # num_stages: 전체 단계 수 (스케줄러의 stage_queue 길이와 동일)
        # Staged-Prefill 모드에서만 사용됨
        self.num_stages = -1
        
        # intermediate_outputs: 각 단계에서 생성된 중간 출력
        # (hidden_states, residual) 튜플 형태로 저장
        # 다음 단계에서 이전 단계의 출력을 재사용하기 위해 저장
        # 메모리 효율성을 위해 마지막 단계 완료 후 None으로 설정
        self.intermediate_outputs = None

    def __len__(self):
        """시퀀스의 전체 토큰 수 반환"""
        return self.num_tokens

    def __getitem__(self, key):
        """인덱싱을 통한 토큰 접근"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """시퀀스가 완료되었는지 확인"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """생성된 토큰 수 (프롬프트 제외)"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def generated_from_last(self):
        """
        마지막 출력 이후 새로 생성된 토큰들 반환
        스트리밍 출력에서 사용됨
        """
        outputs = self.token_ids[self.num_generated_from_last:]
        self.num_generated_from_last = len(self.token_ids)
        return outputs

    @property
    def prompt_token_ids(self):
        """프롬프트 토큰들만 반환"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """생성된 토큰들만 반환 (프롬프트 제외)"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_processed_blocks(self):
        """처리된 블록 수 (KV 캐시 관리용)"""
        return self.num_processed_tokens // self.block_size

    @property
    def num_blocks(self):
        """전체 블록 수 (KV 캐시 관리용)"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """마지막 블록의 토큰 수 (불완전한 블록일 수 있음)"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        특정 블록의 토큰들을 반환
        
        Args:
            i: 블록 인덱스 (0부터 시작)
            
        Returns:
            해당 블록의 토큰 리스트
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        디코딩 단계에서 새로 생성된 토큰을 시퀀스에 추가
        
        Args:
            token_id: 추가할 토큰 ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        시퀀스 직렬화를 위한 상태 반환
        메모리 효율성을 위해 필요한 정보만 저장
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_processed_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        직렬화된 상태에서 시퀀스 복원
        """
        self.num_tokens, self.num_prompt_tokens, self.num_processed_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

    def __repr__(self):
        """
        시퀀스의 문자열 표현 (디버깅용)
        """
        return (f"Sequence(seq_id={self.seq_id}, status={self.status}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_processed_tokens={self.num_processed_tokens}, "
                f"num_tokens_to_process={self.num_tokens_to_process}, block_table={self.block_table}, "
                f"num_completion_tokens={self.num_completion_tokens}, last_token={self.last_token})")

    def __str__(self):
        return self.__repr__()