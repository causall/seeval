from typing import TypeVar, List
from . import data_types as types
Val = TypeVar('Val')


def make_grading_inputs(criteria: types.Criteria, inputs: List[Val]) -> List[types.GradingInput[Val]]:
    grading_inputs = map(lambda val: types.GradingInput[Val](criteria=criteria, input=val),
                         inputs)
    return list(grading_inputs)


def make_grading_inputs_from_eval_dataset(criteria: types.Criteria, eval_dataset: List[types.EvalData[Val]]) -> List[types.GradingInput[Val]]:
    grading_inputs = map(lambda eval_data: types.GradingInput[Val](criteria=criteria, input=eval_data.raw_data),
                         eval_dataset)
    return list(grading_inputs)
