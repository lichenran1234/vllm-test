# This file is the input data parser for the vLLM Uvicorn server.
# It's mostly copied from the Gunicorn server input data parser:
# model-serving/serving-scheduler/serving-resources/mlflow-serving-server/src/mlflowserving/scoring_server/input_data_parser.py

import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict


DF_RECORDS = "dataframe_records"
DF_SPLIT = "dataframe_split"
INSTANCES = "instances"
INPUTS = "inputs"

SUPPORTED_FORMATS = set([DF_RECORDS, DF_SPLIT, INSTANCES, INPUTS])

SUPPORTED_SAMPLING_PARAMS = ["temperature", "max_tokens", "stop", "candidate_count", "top_p"]

REQUIRED_INPUT_FORMAT = "The input schema should either be in OpenAI format or conform to the documentation here: https://docs.databricks.com/en/machine-learning/model-serving/llm-optimized-model-serving.html#input-and-output-schema-format"


# See these docs for the Optimized LLM Inference input format:
# https://docs.google.com/document/d/14i9iuYhcn5CA4pVlXyH72BgTTOzYlDMm6B3hX1KZhRc/edit
# https://docs.google.com/document/d/1eSk5lnawHE3squhgCrQyVlcwbnl-8lrZ75ePArCvCjI/edit
def read_input_data(request_body):
    json_input_str = request_body.decode("utf-8")

    try:
        decoded_input = json.loads(json_input_str)
    except json.decoder.JSONDecodeError as ex:
        raise Exception(f"Invalid input. Ensure that input is a valid JSON formatted string. Error: '{ex}'")

    should_use_open_ai_format = False
    if isinstance(decoded_input, dict):
        format_keys = set(decoded_input.keys()).intersection(SUPPORTED_FORMATS)
        if len(format_keys) > 1:
            message = f"Received dictionary with input fields: {list(format_keys)}"
            raise Exception(f"Invalid input. {message}. {REQUIRED_INPUT_FORMAT}.")
        elif len(format_keys) == 0:
            # No format keys (i.e. dataframe_records, dataframe_split, instances, inputs).
            # So we expect OpenAI input format.
            # See this doc for the decision on OpenAI format:
            # https://docs.google.com/document/d/1eSk5lnawHE3squhgCrQyVlcwbnl-8lrZ75ePArCvCjI/edit
            should_use_open_ai_format = True
        else:
            format = format_keys.pop()
            if format in (INSTANCES, INPUTS):
                filtered_input = {format: decoded_input[format]}
                prompt = _parse_tf_serving_input(filtered_input)
            elif format == DF_SPLIT:
                prompt = _dataframe_from_parsed_json(decoded_input[DF_SPLIT], pandas_orient="split")
            elif format == DF_RECORDS:
                prompt = _dataframe_from_parsed_json(decoded_input[DF_RECORDS], pandas_orient="records")
    elif isinstance(decoded_input, list):
        raise Exception(f"Invalid input. Received a list. {REQUIRED_INPUT_FORMAT}.")
    else:
        message = f"Received unexpected input type '{type(decoded_input)}'"
        raise Exception(f"Invalid input. {message}. {REQUIRED_INPUT_FORMAT}.")

    if should_use_open_ai_format:
        if "prompt" not in decoded_input:
            raise Exception(f"Invalid input. {REQUIRED_INPUT_FORMAT}.")
        prompt = decoded_input.pop("prompt")
        if isinstance(prompt, list):
            pass
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            raise Exception(f"Invalid input. 'prompt' should either be an str or a list. {REQUIRED_INPUT_FORMAT}.")

        sampling_params = {}
        for param, default_value in [("temperature", 0.001), ("max_tokens", 100), ("n", 1), ("top_p", 1)]:
            sampling_params[param] = decoded_input.pop(param, default_value)
        if 'stop' in decoded_input and decoded_input['stop'] is not None:
            # Handle 'stop' with special logic.
            sampling_params['stop'] = list(map(str, decoded_input['stop']))
        return prompt, sampling_params, True


    decoded_input.pop(format)
    params = decoded_input.pop("params", {})
    if not isinstance(params, dict):
        message = f"'params' should be a dict rather than a '{type(params)}'"
        raise Exception(f"Invalid input. {message}. {REQUIRED_INPUT_FORMAT}.")

    if decoded_input:
        message = f"Got unexpected keys {list(decoded_input.keys())} in the input dictionary"
        raise Exception(f"Invalid input. {message}. {REQUIRED_INPUT_FORMAT}.")

    if isinstance(prompt, pd.DataFrame):
        prompt = prompt.to_dict(orient="series")
    if 'prompt' not in prompt:
        raise Exception(f"Invalid input. Missing required field 'prompt'. {REQUIRED_INPUT_FORMAT}.")

    sampling_params_from_legacy_input = {}
    if len(prompt) > 1:
        # Customer is specifying sampling params using one of {dataframe_records, dataframe_split, instances, inputs}
        # It's no longer a valid input format after GPU public preview, but we still support it for backward
        # compatibility. And for simplicity, for batched input, we only use the sampling params for the first data
        # point and apply it to the whole batch.
        for key, value in prompt.items():
            if key in SUPPORTED_SAMPLING_PARAMS:
                sampling_params_from_legacy_input[key] = value.tolist()[0]

    # This is where the sampling params should be parsed from.
    final_sampling_params = {}
    for p in SUPPORTED_SAMPLING_PARAMS:
        if p in params:
            final_sampling_params[p] = params.pop(p)
        elif p in sampling_params_from_legacy_input:
            # Only get sampling params from legacy_input if it doesn't exist in the "params" dict.
            # "Sampling params in the 'params' dict" is the new input format launched with GPU public
            # preview, so it should take precedence.
            final_sampling_params[p] = sampling_params_from_legacy_input[p]

    if params:
        message = (
            f"'params' dict contains unexpected params '{list(params.keys())}'. "
            f"Currently only {SUPPORTED_SAMPLING_PARAMS} are supported"
        )
        raise Exception(f"Invalid input. {message}. {REQUIRED_INPUT_FORMAT}.")

    if 'stop' in final_sampling_params and final_sampling_params['stop'] is not None:
        # Preprocess the 'stop' list. Because VLLM will go into an unrecoverable failed state
        # if the 'stop' list contains non-str elements. We should also consider returning
        # 'invalid input' error when there are non-str elements in the 'stop' list.
        final_sampling_params['stop'] = list(map(str, final_sampling_params['stop']))

    if 'candidate_count' in final_sampling_params:
        final_sampling_params['n'] = final_sampling_params['candidate_count']
        del final_sampling_params['candidate_count']
    else:
        final_sampling_params['n'] = 1

    if 'max_tokens' not in final_sampling_params:
        final_sampling_params['max_tokens'] = 100
    if 'temperature' not in final_sampling_params:
        final_sampling_params['temperature'] = 0.001

    return prompt['prompt'], final_sampling_params, False


def _parse_tf_serving_input(inp_dict):
    """
    :param inp_dict: A dict deserialized from a JSON string formatted as described in TF's
                     serving API doc
                     (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
    """

    def cast_schema_type(input_data):
        if isinstance(input_data, dict):
            return {k: np.array(v) for k, v in input_data.items()}
        return np.array(input_data)

    if "signature_name" in inp_dict:
        raise Exception("Invalid input. 'signature_name' parameter is currently not supported.")

    if not (list(inp_dict.keys()) == ["instances"] or list(inp_dict.keys()) == ["inputs"]):
        raise Exception("Invalid input. One of 'instances' and 'inputs' must be specified (not both or any other keys).")

    if "instances" in inp_dict:
        items = inp_dict["instances"]
        if len(items) > 0 and isinstance(items[0], dict):
            # convert items to column format (map column/input name to tensor)
            data = defaultdict(list)
            for item in items:
                for k, v in item.items():
                    data[k].append(v)
            data = cast_schema_type(data)
        else:
            data = cast_schema_type(items)
    else:
        items = inp_dict["inputs"]
        data = cast_schema_type(items)

    # Sanity check inputted data. This check will only be applied when the row-format `instances`
    # is used since it requires same 0-th dimension for all items.
    if isinstance(data, dict) and "instances" in inp_dict:
        # ensure all columns have the same number of items
        expected_len = len(list(data.values())[0])
        if not all(len(v) == expected_len for v in data.values()):
            raise Exception("Invalid input. The length of values for each input/column name are not the same.")

    return data


def _dataframe_from_parsed_json(decoded_input, pandas_orient):
    """
    Convert parsed json into pandas.DataFrame.
    :param decoded_input: Parsed json - either a list or a dictionary.
    :param pandas_orient: pandas data frame convention used to store the data.
    :return: pandas.DataFrame.
    """

    if pandas_orient == "records":
        if not isinstance(decoded_input, list):
            typemessage = "dictionary" if isinstance(decoded_input, dict) else f"type {type(decoded_input)}"
            raise Exception(f"Invalid input. Dataframe records format must be a list of records. Got {typemessage}.")
        try:
            pdf = pd.DataFrame(data=decoded_input)
        except Exception as ex:
            raise Exception(
                "Invalid input. Provided dataframe_records field is not a valid dataframe "
                f"representation in 'records' format. Error: '{ex}'"
            )
    elif pandas_orient == "split":
        if not isinstance(decoded_input, dict):
            typemessage = "list" if isinstance(decoded_input, list) else f"type {type(decoded_input)}"
            raise Exception(f"Invalid input. Dataframe split format must be a dictionary. Got {typemessage}.")
        keys = set(decoded_input.keys())
        missing_data = "data" not in keys
        extra_keys = keys.difference({"columns", "data", "index"})
        if missing_data or extra_keys:
            raise Exception(
                "Invalid input. Dataframe split format must have 'data' field and optionally 'columns' "
                f"and 'index' fields. Got {keys}.'"
            )
        try:
            pdf = pd.DataFrame(
                index=decoded_input.get("index"),
                columns=decoded_input.get("columns"),
                data=np.array(decoded_input["data"], dtype="object"),
            )
        except Exception as ex:
            raise Exception(
                "Invalid input. Provided dataframe_split field is not a valid dataframe representation in "
                f"'split' format. Error: '{ex}'"
            )
    return pdf
