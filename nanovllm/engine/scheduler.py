from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()
        self.decoding: deque[Sequence] = deque()
        self.schedule_mode = config.schedule_mode

        # Staged-Prefill 모드일 때만 단계별 큐 생성
        # 각 단계마다 별도의 큐를 만들어 시퀀스를 단계별로 관리
        if self.schedule_mode == "staged-prefill":
            # config.num_stages 개수만큼의 단계별 큐 생성
            # 예: num_stages=4이면 stage_queue[0], stage_queue[1], stage_queue[2], stage_queue[3] 생성
            self.stage_queue: list[deque[Sequence]] = [deque() for _ in range(config.num_stages)]

    def is_finished(self):
        return (
            not self.waiting
            and not self.prefilling
            and not self.decoding
        )

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        if self.schedule_mode == "chunked-prefill":
            return self.chunked_prefill_schedule()
        elif self.schedule_mode == "orca":
            return self.orca_schedule()
        elif self.schedule_mode == "staged-prefill":
            return self.staged_prefill_schedule()
        else:
            raise ValueError(f"Unknown schedule mode: {self.schedule_mode}")

    def chunked_prefill_schedule(self) -> list[Sequence]:
        # prefill
        prefill_scheduled_seqs = []
        decode_scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while (
            self.prefilling
            and num_seqs < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.prefilling[0]
            num_seqs += 1
            num_tokens_to_process = min(
                len(seq) - seq.num_processed_tokens,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            num_batched_tokens += num_tokens_to_process
            seq.num_tokens_to_process = num_tokens_to_process

            self.prefilling.popleft()
            prefill_scheduled_seqs.append(seq)

        while (
            self.waiting
            and num_seqs < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)

            num_tokens_to_process = min(
                len(seq) - seq.num_processed_tokens,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            num_batched_tokens += num_tokens_to_process
            seq.num_tokens_to_process = num_tokens_to_process

            self.waiting.popleft()
            prefill_scheduled_seqs.append(seq)

        # decode
        while (
            self.decoding
            and num_seqs < self.max_num_seqs
        ):
            seq = self.decoding.popleft()
            while not self.block_manager.can_append(seq):
                if self.decoding:
                    self.preempt(self.decoding.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                decode_scheduled_seqs.append(seq)

        if prefill_scheduled_seqs:
            for seq in prefill_scheduled_seqs:
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)

        if decode_scheduled_seqs:
            self.decoding.extendleft(reversed(decode_scheduled_seqs))

        scheduled_seqs = prefill_scheduled_seqs + decode_scheduled_seqs

        return scheduled_seqs

    def orca_schedule(self) -> list[Sequence]:
        # prefill
        prefill_scheduled_seqs = []
        decode_scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while (
            self.waiting
            and num_seqs < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting[0]
            if (
                not self.block_manager.can_allocate(seq)
                or len(seq) - seq.num_processed_tokens > self.max_num_batched_tokens - num_batched_tokens
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)

            num_tokens_to_process = len(seq) - seq.num_processed_tokens
            num_batched_tokens += num_tokens_to_process
            seq.num_tokens_to_process = num_tokens_to_process

            self.waiting.popleft()
            prefill_scheduled_seqs.append(seq)

        # decode
        while (
            self.decoding
            and num_seqs < self.max_num_seqs
        ):
            seq = self.decoding.popleft()
            while not self.block_manager.can_append(seq):
                if self.decoding:
                    self.preempt(self.decoding.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                decode_scheduled_seqs.append(seq)

        if prefill_scheduled_seqs:
            for seq in prefill_scheduled_seqs:
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)

        if decode_scheduled_seqs:
            self.decoding.extendleft(reversed(decode_scheduled_seqs))

        scheduled_seqs = prefill_scheduled_seqs + decode_scheduled_seqs

        return scheduled_seqs

    def staged_prefill_schedule(self) -> list[Sequence]:
        """
        Staged-Prefill 스케줄링: 프롬프트 처리를 여러 단계로 나누어 처리
        
        핵심 아이디어:
        1. 각 단계에서 하나의 시퀀스만 처리하여 메모리 사용량 제어
        2. 단계별로 중간 출력을 저장하고 재사용
        3. 마지막 단계 완료 후 디코딩 단계로 전환
        """
        # prefill
        prefill_scheduled_seqs = []
        decode_scheduled_seqs = []
        num_seqs = 0

        # 1단계: 각 단계별 큐에서 하나씩 시퀀스 선택하여 처리
        # 각 단계에서 최대 하나의 시퀀스만 처리하여 메모리 사용량 제어
        for stage in range(len(self.stage_queue)):
            if (
                self.stage_queue[stage]  # 해당 단계에 시퀀스가 있는지 확인
            ):
                seq = self.stage_queue[stage][0]  # 해당 단계의 첫 번째 시퀀스 선택
                # 전체 프롬프트를 한 번에 처리 (청크로 나누지 않음)
                num_tokens_to_process = len(seq) - seq.num_processed_tokens
                seq.num_tokens_to_process = num_tokens_to_process

                self.stage_queue[stage].popleft()  # 선택된 시퀀스를 큐에서 제거
                prefill_scheduled_seqs.append(seq)  # 처리할 시퀀스 목록에 추가
                break  # 하나의 시퀀스만 처리하고 종료

        # 2단계: 새로운 대기 시퀀스를 첫 번째 단계에 추가
        # 대기 중인 새로운 시퀀스가 있고, 시퀀스 수 제한을 넘지 않으면
        if (
            self.waiting
            and num_seqs < self.max_num_seqs
        ):
            seq = self.waiting[0]  # 대기 큐의 첫 번째 시퀀스
            seq.stage = -1  # 새로운 시퀀스는 stage -1로 시작
            seq.num_stages = len(self.stage_queue)  # 전체 단계 수 설정
            if self.block_manager.can_allocate(seq):  # 메모리 블록 할당 가능한지 확인
                self.block_manager.allocate(seq)  # 메모리 블록 할당

                # 전체 프롬프트를 한 번에 처리
                num_tokens_to_process = len(seq) - seq.num_processed_tokens
                seq.num_tokens_to_process = num_tokens_to_process

                self.waiting.popleft()  # 대기 큐에서 제거
                prefill_scheduled_seqs.append(seq)  # 처리할 시퀀스 목록에 추가

        # 3단계: 디코딩 중인 시퀀스들 처리
        # 프롬프트 처리가 완료되어 디코딩 단계에 있는 시퀀스들 처리
        while (
            self.decoding
            and num_seqs < self.max_num_seqs
        ):
            seq = self.decoding.popleft()  # 디코딩 큐에서 시퀀스 가져오기
            # 메모리 블록에 새로운 토큰을 추가할 수 있는지 확인
            while not self.block_manager.can_append(seq):
                if self.decoding:
                    # 다른 디코딩 시퀀스가 있으면 해당 시퀀스를 선점
                    self.preempt(self.decoding.pop())
                else:
                    # 다른 디코딩 시퀀스가 없으면 현재 시퀀스를 선점
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)  # 메모리 블록에 토큰 추가 예약
                decode_scheduled_seqs.append(seq)  # 디코딩 시퀀스 목록에 추가

        # 4단계: 상태 업데이트 및 단계별 큐 관리
        if prefill_scheduled_seqs:
            for seq in prefill_scheduled_seqs:
                seq.status = SequenceStatus.PREFILLING  # 상태를 PREFILLING으로 변경
                seq.stage += 1  # 단계를 1 증가 (예: -1 -> 0, 0 -> 1, ...)
                # 다음 단계가 있으면 해당 단계 큐에 추가
                if seq.stage < len(self.stage_queue):
                    self.stage_queue[seq.stage].append(seq)

        # 디코딩 시퀀스들을 디코딩 큐의 앞쪽에 추가 (우선순위 부여)
        if decode_scheduled_seqs:
            self.decoding.extendleft(reversed(decode_scheduled_seqs))

        scheduled_seqs = prefill_scheduled_seqs + decode_scheduled_seqs

        return scheduled_seqs


    def preempt(self, seq: Sequence):
        """
        시퀀스 선점: 메모리 부족 시 시퀀스를 대기 상태로 되돌림
        """
        print(f"Preempting sequence {seq.seq_id} with status {seq.status}")
        seq.status = SequenceStatus.WAITING  # 상태를 WAITING으로 변경
        seq.stage = -1  # 단계를 -1로 초기화
        seq.intermediate_outputs = None  # 중간 출력 삭제
        self.block_manager.deallocate(seq)  # 메모리 블록 해제
        self.waiting.appendleft(seq)  # 대기 큐의 앞쪽에 추가 (우선순위 부여)

    def postprocess(
            self,
            seqs: list[Sequence],
            token_ids: list[int],
            ) -> list[bool]:
        """
        스케줄링 후 실행된 시퀀스들의 상태를 업데이트하는 후처리 로직
        """
        for seq, token_id in zip(seqs, token_ids):
            if seq.status == SequenceStatus.PREFILLING:
                if self.schedule_mode == "staged-prefill":
                    # Staged-Prefill 모드에서 마지막 단계를 완료한 경우
                    if seq.stage == len(self.stage_queue) - 1:
                        seq.status = SequenceStatus.DECODING  # 상태를 DECODING으로 변경
                        self.stage_queue[seq.stage].remove(seq)  # 마지막 단계 큐에서 제거
                        seq.stage += 1  # 단계를 1 증가 (디코딩 단계로)
                        self.decoding.append(seq)  # 디코딩 큐에 추가
                        seq.num_processed_tokens += seq.num_tokens_to_process  # 처리된 토큰 수 업데이트
                else:
                    # Chunked-Prefill 또는 Orca 모드
                    seq.num_processed_tokens += seq.num_tokens_to_process

                    if seq.num_processed_tokens >= seq.num_prompt_tokens:
                        seq.status = SequenceStatus.DECODING
                        self.prefilling.remove(seq)
                        self.decoding.append(seq)

            if seq.status == SequenceStatus.DECODING:
                seq.append_token(token_id)  # 생성된 토큰을 시퀀스에 추가
                # EOS 토큰이 생성되거나 최대 토큰 수에 도달하면 완료
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED  # 상태를 FINISHED로 변경
                    self.block_manager.deallocate(seq)  # 메모리 블록 해제
                    self.decoding.remove(seq)  # 디코딩 큐에서 제거
