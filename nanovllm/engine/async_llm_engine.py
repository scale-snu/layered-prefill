import time
import logging
import asyncio
from functools import partial
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import multiprocessing as mp

import torch
import nvtx
import numpy as np
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.core_client import CoreClient
from nanovllm.utils.utils import disable_gc

ENGINE_ITERATION_TIMEOUT_S = 600000

logger = logging.getLogger("async_llm_engine")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class TokenizerProcessor(mp.Process):
    def __init__(self, model: str, input_queue: mp.Queue, output_queue: mp.Queue):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        while True:
            item = self.input_queue.get()
            if item is None:
                break
            request_id, prompt, sampling_params = item
            if isinstance(prompt, str):
                token_ids = tokenizer.encode(prompt)
            else:
                token_ids = prompt
            self.output_queue.put((request_id, token_ids, sampling_params))


class AsyncStream:
    """A stream of completed sequence outputs for asynchronous iteration."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[Tuple[str, List[int]], Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> Tuple[str, List[int]]:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    def __init__(self, model: str) -> None:
        self._streams: Dict[str, AsyncStream] = {}
        self._new_in: mp.Queue = mp.Queue()
        self._new_out: mp.Queue = mp.Queue()
        self._finished: asyncio.Queue = asyncio.Queue()
        self._tokenizer_process = TokenizerProcessor(
            model=model,
            input_queue=self._new_in,
            output_queue=self._new_out,
        )
        self._tokenizer_process.start()

    def add_request(self, request_id: str, prompt: Union[str, List[int]], sampling_params: SamplingParams) -> AsyncStream:
        if request_id in self._streams:
            raise KeyError(f"Request {request_id} already exists.")
        stream = AsyncStream(request_id)
        self._streams[request_id] = stream
        self._new_in.put_nowait((request_id, prompt, sampling_params))
        return stream

    def abort_request(self, request_id: str) -> None:
        assert request_id in self._streams, f"Request {request_id} not found."
        self._finished.put_nowait(request_id)
        self._streams[request_id].finish()

    def get_new_and_finished(self) -> Tuple[List[Tuple[AsyncStream, str, Union[str, List[int]], SamplingParams]], Set[str]]:
        new = []
        finished: Set[str] = set()

        while not self._finished.empty():
            finished.add(self._finished.get_nowait())

        while not self._new_out.empty():
            rid, prompt, sp = self._new_out.get_nowait()
            stream = self._streams[rid]
            if rid in finished:
                stream.finish()
                # del self._streams[rid]
            else:
                new.append((stream, rid, prompt, sp))

        return new, finished


def _log_task_completion(
    task: asyncio.Task, error_callback: Callable[[Exception], None]
) -> None:
    try:
        rv = task.result()
        raise AssertionError(f"Background loop should not finish without error: {rv}")
    except asyncio.CancelledError:
        return
    except Exception as e:
        error_callback(e)
        raise RuntimeError("Engine loop failed.") from e


class _AsyncLLMEngine:
    def __init__(self, config: Config) -> None:
        # instantiate workers
        self.ps = []
        self.events = []
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.vocab = self.tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.vocab.items()}
        config.eos = self.tokenizer.eos_token_id
        self.core_client = CoreClient.make_core_client(config)
        logger.info("LLMEngine init")

    def add_request(self, seq_id: str, prompt: Union[str, List[int]], sampling_params: SamplingParams) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, seq_id)
        self.core_client.add_request(seq)

    @disable_gc()
    def step(self) -> List[Tuple[str, List[int], str, bool]]:
        seqs = self.core_client.get_output()
        if not seqs:
            return []
        generated_from_last = [[seq.last_token if seq.last_token >= 0 else 0] for seq in seqs]
        generated_from_last_dec = self.tokenizer.batch_decode(generated_from_last, skip_special_tokens=True)
        ret = []
        for seq, gen_dec, gen in zip(seqs, generated_from_last_dec, generated_from_last):
            if seq.status == SequenceStatus.ERROR:
                ret.append((False, (seq.seq_id, seq.error_message, [], True)))
                continue
            if seq.status not in [SequenceStatus.DECODING, SequenceStatus.FINISHED]:
                ret.append((True, None))
                continue
            ret.append((
                True,
                (
                    seq.seq_id,
                    gen_dec,
                    gen,
                    seq.is_finished,
                ),
            ))
        return ret


class AsyncLLMEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine = _AsyncLLMEngine(config)
        self._tracker: RequestTracker
        self._loop_task: Optional[asyncio.Task] = None
        self._errored: Optional[Exception] = None

    def start(self) -> None:
        if self._loop_task and not self._loop_task.done():
            return
        self._tracker = RequestTracker(self.config.model)
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())
        self._loop_task.add_done_callback(partial(_log_task_completion, error_callback=self._on_error))

    def _on_error(self, exc: Exception) -> None:
        self._errored = exc
        # abort all
        for rid in list(self._tracker._streams.keys()):
            self._tracker.abort_request(rid)

    @disable_gc()
    async def _engine_step(self) -> bool:
        s = time.time()
        new, finished = self._tracker.get_new_and_finished()
        for stream, rid, prompt, sp in new:
            try:
                self._engine.add_request(rid, prompt, sp)
            except Exception as e:
                stream.put(e)
                stream.finish()
        for rid in finished:
            # no direct abort support in scheduler
            pass
        m = time.time()
        ret = await asyncio.get_event_loop().run_in_executor(None, self._engine.step)
        for is_good, outputs in ret:
            if outputs is None:
                continue

            if not is_good:
                rid, error_message, _, _ = outputs
                assert rid in self._tracker._streams, f"Stream {rid} not found in tracker."
                self._tracker._streams[rid].put(Exception(error_message))
                self._tracker._streams[rid].finish()
                continue

            rid, generated_text, tokens, is_finished = outputs

            assert rid in self._tracker._streams, f"Stream {rid} not found in tracker."
            self._tracker._streams[rid].put((rid, generated_text, tokens))
            if is_finished:
                self._tracker.abort_request(rid)
        e = time.time()
        duration = e - s
        if duration > 0.08:
            logger.warning(f"Engine async step took too long: {duration:.3f}s")
            logger.info(f"  Tracker processing time: {m - s:.3f}s"
                        f"  Engine step time: {e - m:.3f}s")
        return True

    async def _run_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._engine_step(), ENGINE_ITERATION_TIMEOUT_S)
            except asyncio.TimeoutError as e:
                self._on_error(e)
                raise
            await asyncio.sleep(0)

    async def generate(
        self,
        request_id: str,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[Tuple[str, List[int]]]:
        if not self._loop_task or self._loop_task.done():
            self.start()
        stream = self._tracker.add_request(request_id, prompt, sampling_params)
        try:
            async for item in stream:
                yield item
        except Exception:
            self._tracker.abort_request(request_id)
            # raise

    async def exit(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        self._engine.exit()
