import dspy
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import data_types as types


def run_parallel[T, R](module: types.ForwardModule[T, R],
                       args_list: List[T],
                       lm: dspy.LM,
                       concurrency: int) -> types.ResponseData[R]:
    def executor(args: T):
        with dspy.context(lm=lm):
            # if is a dict or typedict **unpack it otherwise pass as is
            if isinstance(args, dict):
                result = module(**args)
            else:
                result = module(args)
            return module.get_value(result), result

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(executor, args): i for i,
                   args in enumerate(args_list)}
        results: List[R | None] = [None] * len(args_list)
        preds: List[dspy.Prediction | None] = [None] * len(args_list)
        # thread-safe because they write to different areas of memory
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx], preds[idx] = fut.result()
    return types.ResponseData(data=results, debug=preds)
