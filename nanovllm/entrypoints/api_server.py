"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import json
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.config import APIServerConfig


TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    model = request_dict.pop("model", None)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(request_id, prompt, sampling_params)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = request_output[1]
            token_ids = request_output[2]
            ret = {"generated_text": text_outputs, "output_tokens": token_ids}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = prompt + final_output.text
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    config = APIServerConfig(
        # model="/data3/cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9/",
        model="/data3/cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/d47b0d4ae4b48fde975756bf360a63a9cca8d470/",
        # model="/data3/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/",
        max_num_batched_tokens=512,
        max_num_seqs=64,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enforce_eager=False,
        log_level="debug",
        host="localhost",
        port=8000,
        nccl_port=2333,
        schedule_mode="staged-prefill",  # or "orca" or "staged-prefill"
        num_stages=4,
    )

    # Create the system config from the config.
    engine = AsyncLLMEngine(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        workers=1,
    )
