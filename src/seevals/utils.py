import pydantic
from typing import Dict, Type, List, TypeVar, Optional
from . import data_types as types
import json
import dspy
from typing import Callable
import numpy as np
import math
import os
from scipy.stats import multivariate_hypergeom
from itertools import product
from itertools import combinations, chain
from jsonpath_ng import parse
from math import comb


def for_each_weak_composition(N: int, k: int, consume):
    """Stars & bars; calls `consume(view)` for each k-tuple.
       `consume` MUST copy if it needs to keep the data."""
    if k == 1:
        buf = [N]
        consume(buf)
        return
    L = N + k - 1
    buf = [0]*k
    last = (L,)  # sentinel

    for bars in combinations(range(L), k-1):  # C-level iterator
        prev = -1
        i = 0
        for b in chain(bars, last):
            buf[i] = b - prev - 1
            prev, i = b, i + 1
        consume(buf)  # no tuple allocation


def dump_to_memmap(N: int, k: int, path: str, dtype=np.uint8) -> str:
    total = math.comb(N + k - 1, k - 1)
    arr = np.memmap(path, mode='w+', dtype=dtype, shape=(total, k))
    row = 0

    def consume(buf):
        nonlocal row
        arr[row, :] = buf  # fast C-level copy of small k
        row += 1
    for_each_weak_composition(N, k, consume)
    arr.flush()
    return path


def iter_weak_compositions(N: int, k: int):
    """
    Yield all ordered k-tuples (x1,...,xk) of nonnegative ints with sum N.
    Stars & bars: choose k-1 bar positions among N+k-1 slots, then
    counts are gaps between consecutive bars (with sentinels).
    this allows us to then put samples and the possible distributions of those samples into proper
    buckets so we can enumerate over all the possible combinations to create an eventual confidence
    region for the distribution of the samples.
    """
    L = N + k - 1                    # total slots = stars + bars
    if N < 0 or k <= 0:
        return
    if k == 1:
        yield (N,)
        return

    # bar indices in [0, L-1], sorted
    for bars in combinations(range(L), k - 1):
        prev = -1
        comp = []
        for b in bars + (L,):                   # append sentinel "bar" at the end
            # gap size = #stars in this bucket
            comp.append(b - prev - 1)
            prev = b
        yield tuple(comp)


def weak_compositions_array(N: int, k: int) -> np.ndarray:
    return np.asarray(list(iter_weak_compositions(N, k)), dtype=int)

# sample size , number of discrete classes, overall total

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


def write_eval_dataset(path: str, eval_dataset: List[types.EvalData[LT]]):
    with open(path, 'w') as f:
        for eval_data in eval_dataset:
            f.write(eval_data.model_dump_json()+'\n')


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

# here we use the multivariate hypergeometric distribution to calculate the coverage of an interv
# here we produce a coverage region for the likelihood of a sample being produced by a configuration of the population
# we then use this coverage region to produce a confidence interval for the mean being in the coverage region  for a % number of times if we drew more samples

# def calc_multivariate_hypergeometric_coverage(num_samples: int, population_size: int, upper_bound: float, lower_bound: float, confidence: float) -> :
#    sigma = 1 - confidence


def calc_multivariate_pmf(sample: List[int]):
    result = multivariate_hypergeom.pmf(x=[1, 2, 3], m=[10, 10, 10], n=6)
    print(result)


def make_sample_criteria(num_samples: int, upper_bound: float, lower_bound: float, confidence: float) -> types.SampleCriteria:
    return SampleCriteria(
        num_samples=num_samples,
        confidence=confidence,
        error_margin=get_hoeffding_error_margin(confidence, num_samples)
    )
