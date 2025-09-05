import pydantic
from typing import Type, List, TypeVar, Optional
import data_types as types
import json
import dspy
from typing import Callable
import numpy as np

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


def get_values(values: List[types.R | None]) -> List[types.R]:
    return [value for value in values if value is not None]


LT = TypeVar('LT', bound=pydantic.BaseModel)


def load_from_result(path: str, LoadingType: Type[LT]) -> List[LT]:
    return load_from(path, LoadingType, "item")


def load_from(path: str, LoadingType: Type[LT], prefix: str | None = None) -> List[LT]:
    results: List[LT] = []
    with open(path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if prefix is not None:
                results.append(LoadingType.model_validate(
                    result.get(prefix, {})))
            else:
                results.append(LoadingType.model_validate(result))
    return results


VV = TypeVar('VV', bound='pydantic.BaseModel')


def write_results_from_response(path: str, results: types.ResponseData[LT], criteria: Optional[types.Criteria] = None):
    with open(path, 'w') as f:
        for result in get_values(results.data):
            if criteria is not None:
                f.write(json.dumps(
                    {'item': result.model_dump(), 'criteria': criteria.model_dump()}) + '\n')
            else:
                f.write(json.dumps({'item': result.model_dump()}) + '\n')


def calc_hoeffding_error(num_samples: int, upper_bound: float, lower_bound: float, confidence: float) -> float:
    sigma = 1 - confidence
    return np.sqrt(np.pow((upper_bound-lower_bound), 2) /
                   (2 * num_samples)) * np.log(2/(sigma))


def calc_serfling_error(num_samples: int, population_size: int, upper_bound: float, lower_bound: float, confidence: float) -> float:
    if (num_samples > population_size):
        return np.nan
    sigma = 1 - confidence
    fpc = 1.0 - ((num_samples - 1)/population_size)
    # piecewise function for fpc Theorem 2.4 and Corollary 2.5. https://arxiv.org/pdf/1309.4029
    if num_samples >= population_size/2.0:
        fpc = (1.0 - num_samples/population_size) * (1 + 1/num_samples)

    return (upper_bound-lower_bound)*np.sqrt(fpc / (2 * num_samples) * np.log(2.0/(sigma)))


def make_sample_criteria(num_samples: int, upper_bound: float, lower_bound: float, confidence: float) -> types.SampleCriteria:
    return SampleCriteria(
        num_samples=num_samples,
        confidence=confidence,
        error_margin=get_hoeffding_error_margin(confidence, num_samples)
    )
