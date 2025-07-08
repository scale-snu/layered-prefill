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

from transformers import AutoTokenizer

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config

ENGINE_ITERATION_TIMEOUT_S = 600


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
    def __init__(self) -> None:
        self._streams: Dict[str, AsyncStream] = {}
        self._new: asyncio.Queue = asyncio.Queue()
        self._finished: asyncio.Queue = asyncio.Queue()
        self.new_event = asyncio.Event()

    def add_request(self, request_id: str, prompt: Union[str, List[int]], sampling_params: SamplingParams) -> AsyncStream:
        if request_id in self._streams:
            raise KeyError(f"Request {request_id} already exists.")
        stream = AsyncStream(request_id)
        self._new.put_nowait((stream, request_id, prompt, sampling_params))
        self.new_event.set()
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

        while not self._new.empty():
            stream, rid, prompt, sp = self._new.get_nowait()
            if rid in finished:
                stream.finish()
            else:
                self._streams[rid] = stream
                new.append((stream, rid, prompt, sp))

        return new, finished

    async def wait_for_new(self):
        if self._new.empty():
            await self.new_event.wait()
        self.new_event.clear()


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
        ctx = __import__("torch").multiprocessing.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            ev = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, i, ev))
            p.start()
            self.ps.append(p)
            self.events.append(ev)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

    def add_request(self, seq_id: str, prompt: Union[str, List[int]], sampling_params: SamplingParams) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params, seq_id))

    def step(self) -> List[Tuple[str, List[int], str, bool]]:
        seqs = self.scheduler.schedule()
        if not seqs:
            return []
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        generated_from_lasts = [seq.generated_from_last for seq in seqs]
        outputs = [
            (
                seq.seq_id,
                self.tokenizer.decode(generated_from_last, skip_special_tokens=True),
                generated_from_last,
                seq.is_finished,
            )
            for seq, generated_from_last in zip(seqs, generated_from_lasts)
            if generated_from_last
        ]
        return outputs

    def exit(self) -> None:
        self.model_runner.call("exit")
        for p in self.ps:
            p.join()


class AsyncLLMEngine:
    def __init__(self, config: Config) -> None:
        self._engine = _AsyncLLMEngine(config)
        self._tracker: RequestTracker
        self._loop_task: Optional[asyncio.Task] = None
        self._errored: Optional[Exception] = None

    def start(self) -> None:
        if self._loop_task and not self._loop_task.done():
            return
        self._tracker = RequestTracker()
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())
        self._loop_task.add_done_callback(partial(_log_task_completion, error_callback=self._on_error))

    def _on_error(self, exc: Exception) -> None:
        self._errored = exc
        # abort all
        for rid in list(self._tracker._streams.keys()):
            self._tracker.abort_request(rid)

    async def _engine_step(self) -> bool:
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
        outputs = await asyncio.get_event_loop().run_in_executor(None, self._engine.step)
        for rid, generated_text, tokens, is_finished in outputs:
            # if rid in self._tracker._streams:
            assert rid in self._tracker._streams, f"Stream {rid} not found in tracker."
            self._tracker._streams[rid].put((rid, generated_text, tokens))
            if is_finished:
                self._tracker.abort_request(rid)
        return bool(outputs)

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
            raise

    async def exit(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        self._engine.exit()
