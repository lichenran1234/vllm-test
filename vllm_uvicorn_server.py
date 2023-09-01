import argparse
import json
import os
import pandas as pd
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from input_data_parser import read_input_data


SUPPORTED_SAMPLING_PARAMS = ["presence_penalty", "frequency_penalty", "temperature", "top_p", "top_k", "stop", "max_tokens"]
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/model/components/tokenizer")
MODEL_BINARY_PATH = os.environ.get("MODEL_BINARY_PATH", "/model/model")


app = FastAPI()


# Use `async def` rather than `def` so this function will be executed in the main
# event loop (together with model invocation requests). Therefore, "OK" response
# from this endpoint indicates the main event loop is healthy.
@app.get("/v2/health/live")
async def live():
    return "OK"

# Use `async def` rather than `def` so this function will be executed in the main
# event loop (together with model invocation requests). Therefore, "OK" response
# from this endpoint indicates the main event loop is healthy.
@app.get("/v2/health/ready")
async def ready():
    return "OK"

@app.post("/invocations")
async def generate(request: Request) -> Response:
    request_body = await request.body()
    
    try:
        model_input = read_input_data(request_body)
        
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="series")
        for key, value in model_input.items():
            # Only take the first value and silently drop the rest. This means
            # only the first prompt takes effect in each HTTP request.
            model_input[key] = value.tolist()[0]
        
        if "prompt" not in model_input:
            raise Exception("Invalid input. Model input missing required field 'prompt'.")
        prompt = model_input.pop("prompt")
        
        sampling_params = {}
        sampling_params["n"] = model_input.pop("candidate_count", 1)
        for p in SUPPORTED_SAMPLING_PARAMS:
            if p in model_input:
                sampling_params[p] = model_input.pop(p)
        
        if model_input:
            raise Exception(
                f"Invalid input. Model input contains unexpected parameters {list(model_input.keys())}."
                f" Only these parameters are supported: {SUPPORTED_SAMPLING_PARAMS + ['prompt', 'candidate_count']}."
            )
        
        # Following are mostly copied from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
        sampling_params = SamplingParams(**sampling_params)
        request_id = random_uuid()
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
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
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output
        if final_output is None:
            raise Exception("Server failed to process this request.")
        
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
        return JSONResponse({"predictions": [{"candidates": candidates, "metadata": metadata}]})
        
    except Exception as ex:
        # Return all errors as 400 for now.
        print(f"Request failed. HTTP response code: 400. Error message: '{ex}'")
        return JSONResponse(status_code=400, content={"error_message": str(ex)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=str, default=MODEL_BINARY_PATH)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=False)
