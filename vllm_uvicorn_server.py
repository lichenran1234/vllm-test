import argparse
import json
import os
import pandas as pd
import time
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from input_data_parser import read_input_data


app = FastAPI()


# Use `async def` rather than `def` so this function will be executed in the main
# event loop (together with model invocation requests). Therefore, "OK" response
# from this endpoint indicates the main event loop is healthy. "Main event loop
# being healthy" is the best approximation of "server being healthy".
@app.get("/v2/health/live")
async def live():
    return "OK"

# Use `async def` rather than `def` so this function will be executed in the main
# event loop (together with model invocation requests). Therefore, "OK" response
# from this endpoint indicates the main event loop is healthy. "Main event loop
# # being healthy" is the best approximation of "server being healthy".
@app.get("/v2/health/ready")
async def ready():
    return "OK"

@app.post("/invocations")
async def generate(request: Request) -> Response:
    request_body = await request.body()

    try:
        # TODO: consider moving this pre-processing logic to an external thread pool
        # or process pool if it's taking too much time.
        prompt, sampling_params, should_use_open_ai_format = read_input_data(request_body)

        # Following are mostly copied from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
        sampling_params = SamplingParams(**sampling_params)

        results_generators = []
        for p in prompt:
            request_id = random_uuid()
            results_generator = engine.generate(p, sampling_params, request_id)
            results_generators.append((request_id, results_generator))

        # Streaming not supported yet
        # async def stream_results() -> AsyncGenerator[bytes, None]:
        #     async for request_output in results_generator:
        #         prompt = request_output.prompt
        #         text_outputs = [
        #             prompt + output.text for output in request_output.outputs
        #         ]
        #         ret = {"text": text_outputs}
        #         yield (json.dumps(ret) + "\0").encode("utf-8")

        # async def abort_request() -> None:
        #     await engine.abort(request_id)

        # if stream:
        #     background_tasks = BackgroundTasks()
        #     # Abort the request if the client disconnects.
        #     background_tasks.add_task(abort_request)
        #     return StreamingResponse(stream_results(), background=background_tasks)

        # Non-streaming case
        outputs = []
        for request_id, results_generator in results_generators:
            final_output = None
            async for request_output in results_generator:
                if await request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await engine.abort(request_id)
                final_output = request_output
            if final_output is None:
                raise Exception(f"Server failed to process this request. Prompt: {prompt}. Sampling params: {sampling_params}")
            outputs.append((request_id, final_output))

        if should_use_open_ai_format:
            # If the input is in OpenAI format, return the output in OpenAI format.
            choices = []
            prompt_tokens = 0
            completion_tokens = 0
            idx = 0
            for request_id, final_output in outputs:
                prompt_tokens += len(final_output.prompt_token_ids)
                for output in final_output.outputs:
                    choices.append({
                        "text": output.text,
                        "index": idx,
                        "finish_reason": output.finish_reason
                    })
                    completion_tokens += len(output.token_ids)
                    idx += 1
            output_json_dict = {
                "id": str(random_uuid()),
                "object": "text_completion",
                "created": int(time.time()),
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            return JSONResponse(output_json_dict)

        predictions = []
        for request_id, final_output in outputs:
            prompt_length = len(final_output.prompt_token_ids)
            candidates = []
            output_length = 0
            for output in final_output.outputs:
                candidates.append({"text": output.text, "metadata": {"finish_reason": output.finish_reason}})
                output_length += len(output.token_ids)
            metadata = {
                "input_tokens": prompt_length,
                "output_tokens": output_length,
                "total_tokens": prompt_length + output_length,
            }
            predictions.append({"candidates": candidates, "metadata": metadata})

        return JSONResponse({"predictions": predictions})

    except Exception as ex:
        # Return all errors as 400 for now.
        return JSONResponse(status_code=400, content={"error_message": str(ex)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=False)
