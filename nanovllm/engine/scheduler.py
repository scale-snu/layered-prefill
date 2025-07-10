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
        raise NotImplementedError("Staged prefill scheduling is not implemented yet.")

    def preempt(self, seq: Sequence):
        print(f"Preempting sequence {seq.seq_id} with status {seq.status}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
            self,
            seqs: list[Sequence],
            token_ids: list[int],
            ) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if seq.status == SequenceStatus.PREFILLING:
                seq.num_processed_tokens += seq.num_tokens_to_process

                if seq.num_processed_tokens >= seq.num_prompt_tokens:
                    seq.status = SequenceStatus.DECODING
                    self.prefilling.remove(seq)
                    self.decoding.append(seq)

            if seq.status == SequenceStatus.DECODING:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.decoding.remove(seq)
