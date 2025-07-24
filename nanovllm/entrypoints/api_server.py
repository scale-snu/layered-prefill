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


def parse_args() -> APIServerConfig:
    """Parse command line arguments and return the config."""
    import argparse

    parser = argparse.ArgumentParser(description="API server for AsyncLLMEngine.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=512,
                        help="Maximum number of tokens to batch together for generation.")
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="Maximum number of sequences to generate in parallel.")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Maximum length of the model input sequence.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for the model.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for the model.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Whether to enforce eager execution mode.")
    parser.add_argument("--log-level", type=str, default="debug",
                        help="Logging level for the server.")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host address for the API server.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the API server.")
    parser.add_argument("--nccl-port", type=int, default=2333,
                        help="NCCL port for distributed training.")
    parser.add_argument("--schedule-mode", type=str, default="staged-prefill",
                        choices=["staged-prefill", "orca", "chunked-prefill"],
                        help="Scheduling mode for the generation.")
    parser.add_argument("--num-stages", type=int, default=2,
                        help="Number of stages for staged prefill scheduling.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    config = APIServerConfig(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        log_level=args.log_level,
        host=args.host,
        port=args.port,
        nccl_port=args.nccl_port,
        schedule_mode=args.schedule_mode,
        num_stages=args.num_stages,
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
