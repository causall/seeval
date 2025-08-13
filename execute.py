import dspy
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import data_types as types



def run_parallel(module: types.ForwardModule[types.T, types.R],
    args_list: List[types.T],
    lm: dspy.LM,
    concurrency: int)-> types.ResponseData[types.R]:
    def executor(args: types.T):
        with dspy.context(lm=lm):
            return module.forward(args)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(executor, args): i for i, args in enumerate(args_list)}
        results: List[types.R | None] = [None] * len(args_list)
        # thread-safe because they write to different areas of memory
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = module.get_value(fut.result())
    return types.ResponseData(data=results)
