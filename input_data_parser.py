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

REQUIRED_INPUT_FORMAT = f"The input must be a JSON dictionary with exactly one of the input fields {SUPPORTED_FORMATS}"

# Payload logging is not yet supported in this Uvicorn vLLM server.
# This key is only needed for properly parsing the input.
PAYLOAD_LOGGING_INFERENCE_ID_KEY = os.environ.get(
    "PAYLOAD_LOGGING_INFERENCE_ID_KEY", "inference_id"
)


def read_input_data(request_body):
    json_input_str = request_body.decode("utf-8")

    try:
        decoded_input = json.loads(json_input_str)
    except json.decoder.JSONDecodeError as ex:
        raise Exception(f"Invalid input. Ensure that input is a valid JSON formatted string. Error: '{ex}'")

    if isinstance(decoded_input, dict):
        format_keys = set(decoded_input.keys()).intersection(SUPPORTED_FORMATS)
        if len(format_keys) != 1:
            message = f"Received dictionary with input fields: {format_keys}"
            raise Exception(f"Invalid input. {REQUIRED_INPUT_FORMAT}. {message}.")
        format = format_keys.pop()
        if format in (INSTANCES, INPUTS):
            # Remove the inference_id from the input dict before passing it to parse_tf_serving_input.
            # Otherwise it will throw an error.
            filtered_input = {
                k: decoded_input[k]
                for k in decoded_input
                if k != PAYLOAD_LOGGING_INFERENCE_ID_KEY
            }
            return _parse_tf_serving_input(filtered_input)
        elif format == DF_SPLIT:
            return _dataframe_from_parsed_json(decoded_input[DF_SPLIT], pandas_orient="split")
        elif format == DF_RECORDS:
            return _dataframe_from_parsed_json(decoded_input[DF_RECORDS], pandas_orient="records")
    elif isinstance(decoded_input, list):
        raise Exception(f"Invalid input. {REQUIRED_INPUT_FORMAT}. Received a list.")

    message = f"Received unexpected input type '{type(decoded_input)}'"
    raise Exception(f"Invalid input. {REQUIRED_INPUT_FORMAT}. {message}.")


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
