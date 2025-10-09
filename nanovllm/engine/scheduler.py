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
        self.max_model_len = config.max_model_len

        if self.schedule_mode == "staged-prefill":
            self.stage_queue: list[deque[Sequence]] = [deque() for _ in range(config.num_stages)]
            self.current_stage = -1

    def is_finished(self):
        return (
            not self.waiting
            and not self.prefilling
            and not self.decoding
        )

    def add(self, seq: Sequence):
        if len(seq) > self.max_model_len:
            raise ValueError(f"Sequence length {len(seq)} exceeds max model length {self.max_model_len}.")
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
        num_batched_tokens = 0
        num_seqs = 0

        # decode
        while (
            self.decoding
            and num_batched_tokens < self.max_num_batched_tokens
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
                num_batched_tokens += 1
                self.block_manager.may_append(seq)
                decode_scheduled_seqs.append(seq)
                num_seqs += 1

        # prefill
        while (
            self.prefilling
            and num_batched_tokens < self.max_num_batched_tokens
            and num_seqs < self.max_num_seqs
        ):
            seq = self.prefilling[0]
            num_tokens_to_process = min(
                len(seq) - seq.num_processed_tokens,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            num_batched_tokens += num_tokens_to_process
            seq.num_tokens_to_process = num_tokens_to_process

            self.prefilling.popleft()
            prefill_scheduled_seqs.append(seq)
            num_seqs += 1

        while (
            self.waiting
            and num_batched_tokens < self.max_num_batched_tokens
            and num_seqs < self.max_num_seqs
        ):
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)

            num_tokens_to_process = min(
                len(seq) - seq.num_processed_tokens,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            num_batched_tokens += num_tokens_to_process
            seq.num_tokens_to_process = num_tokens_to_process

            self.waiting.popleft()
            prefill_scheduled_seqs.append(seq)
            num_seqs += 1

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

    def get_num_stages(self, num_batched_tokens:int, num_attn_tokens: int) -> int:
        # return len(self.stage_queue)
        if len(self.stage_queue) == 16:
            if num_batched_tokens <= 512:
                return 1
            elif num_batched_tokens <= 1024:
                return 2
            elif num_batched_tokens <= 2048:
                return 4
            elif num_batched_tokens <= 4096:
                return 8
            else:
                return 16
        elif len(self.stage_queue) == 4:
            if num_batched_tokens <= 512:
                return 1
            elif num_batched_tokens <= 1024:
                return 2
            else:
                return 4
        elif len(self.stage_queue) == 12:
            if num_batched_tokens <= 512:
                return 1
            elif num_batched_tokens <= 1024:
                return 2
            elif num_batched_tokens <= 2048:
                return 4
            elif num_batched_tokens <= 4096:
                return 6
            else:
                return 12
        elif len(self.stage_queue) == 24:
            if num_batched_tokens <= 512:
                return 1
            elif num_batched_tokens <= 1024:
                return 2
            elif num_batched_tokens <= 2048:
                return 4
            elif num_batched_tokens <= 4096:
                return 8
            else:
                return 24
        else:
            raise ValueError(f"Unsupported number of stages: {len(self.stage_queue)}")

    def staged_prefill_schedule(self) -> list[Sequence]:
        # prefill
        prefill_scheduled_seqs = []
        decode_scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        num_attn_tokens = 0

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

        if self.current_stage >= 0:
            while self.stage_queue[self.current_stage]:
                seq = self.stage_queue[self.current_stage][0]

                self.stage_queue[self.current_stage].popleft()
                self.prefilling.remove(seq)
                prefill_scheduled_seqs.append(seq)

        if not prefill_scheduled_seqs:
            # prefill
            while (
                self.prefilling
                and num_seqs < self.max_num_seqs
                and num_batched_tokens < self.max_num_batched_tokens
            ):
                seq = self.prefilling[0]
                seq.stage = -1
                seq.num_stages = -1

                num_tokens_to_process = min(
                    len(seq) - seq.num_processed_tokens,
                    self.max_num_batched_tokens - num_batched_tokens,
                )
                num_seqs += 1
                seq.num_tokens_to_process = num_tokens_to_process
                num_batched_tokens += num_tokens_to_process
                num_attn_tokens += num_tokens_to_process * (seq.num_processed_tokens + num_tokens_to_process)

                self.prefilling.popleft()
                prefill_scheduled_seqs.append(seq)

            while (
                self.waiting
                and num_seqs < self.max_num_seqs
                and num_batched_tokens < self.max_num_batched_tokens
            ):
                seq = self.waiting[0]
                seq.stage = -1
                seq.num_stages = -1
                if self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)

                    num_tokens_to_process = min(
                        len(seq) - seq.num_processed_tokens,
                        self.max_num_batched_tokens - num_batched_tokens,
                    )

                    num_seqs += 1
                    seq.num_tokens_to_process = num_tokens_to_process
                    num_batched_tokens += num_tokens_to_process
                    num_attn_tokens += num_tokens_to_process * (seq.num_processed_tokens + num_tokens_to_process)

                    self.waiting.popleft()
                    prefill_scheduled_seqs.append(seq)
                else:
                    break

            for seq in prefill_scheduled_seqs:
                seq.num_stages = self.get_num_stages(num_batched_tokens, num_attn_tokens)
            if len(prefill_scheduled_seqs) > 0:
                print(f"num_seqs: {num_seqs}, len(prefill_scheduled_seqs): {len(prefill_scheduled_seqs)}, num_batched_tokens: {num_batched_tokens}, num_attn_tokens: {num_attn_tokens}, num_stages: {prefill_scheduled_seqs[0].num_stages if prefill_scheduled_seqs else 0}")

        if prefill_scheduled_seqs:
            for seq in prefill_scheduled_seqs:
                seq.status = SequenceStatus.PREFILLING
                seq.stage += 1
                if seq.stage < seq.num_stages:
                    self.stage_queue[seq.stage].append(seq)
                    self.current_stage = seq.stage
                else:
                    self.current_stage = -1
                self.prefilling.append(seq)

        if decode_scheduled_seqs:
            self.decoding.extendleft(reversed(decode_scheduled_seqs))

        scheduled_seqs = prefill_scheduled_seqs + decode_scheduled_seqs

        return scheduled_seqs

    def preempt(self, seq: Sequence):
        print(f"Preempting sequence {seq.seq_id} with status {seq.status}")
        seq.status = SequenceStatus.WAITING
        seq.stage = -1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
            self,
            seqs: list[Sequence],
            token_ids: list[int],
            ) -> list[bool]:
        assert len(seqs) == len(token_ids), f"Number of sequences and token IDs must match. ({len(seqs)} != {len(token_ids)})"
        for seq, token_id in zip(seqs, token_ids):
            if seq.status == SequenceStatus.PREFILLING:
                if self.schedule_mode == "staged-prefill":
                    if seq.stage == seq.num_stages - 1:
                        seq.num_processed_tokens += seq.num_tokens_to_process
                        if seq.num_processed_tokens >= len(seq):
                            seq.status = SequenceStatus.DECODING
                            self.stage_queue[seq.stage].remove(seq)
                            seq.stage += 1
                            self.decoding.append(seq)
                            self.prefilling.remove(seq)
                        else:
                            seq.status = SequenceStatus.PREFILLING
                            self.stage_queue[seq.stage].remove(seq)
                else:
                    seq.num_processed_tokens += seq.num_tokens_to_process

                    if seq.num_processed_tokens >= len(seq):
                        seq.status = SequenceStatus.DECODING
                        self.prefilling.remove(seq)
                        self.decoding.append(seq)

            if seq.status == SequenceStatus.DECODING:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens or len(seq) >= self.max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.decoding.remove(seq)
