import pydantic
from typing import Type, List
import data_types as types
import dspy

# create a decorator so we an add formatted instructions to signatures
def instructions(**kwargs):
    def deco(fn):
        fn.__doc__ = fn.__doc__.format(**kwargs)
        return fn
    return deco

def generate_json_output_field(class_schema: Type[pydantic.BaseModel]):
    return dspy.OutputField(
        description=f"A single, valid JSON object. Strictly adhere to the schema provided below. Ensure correct JSON syntax, including proper quoting and escaping. 'json_schema': {class_schema.model_json_schema()}. Prefix the output with ```json a json code block and suffix with ```")

def extract_json_from_code_block(raw: str) -> str:
    """
    Extract the first JSON object from a string, stripping markdown/code block fences and extraneous text.
    Returns the JSON string, or raises ValueError if not found.
    """
    json_start = raw.find('{')
    json_end = raw.rfind('}')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise ValueError(
            f"Could not find valid JSON delimiters in output: {raw[:100]}...")
    return raw[json_start:json_end+1]

def get_values(values: List[types.R|None])->List[types.R]:
        return [value for value in values if value is not None]
