from typing import TypeVar, List
import data_types as types
Val = TypeVar('Val')

def make_grading_inputs(criteria: types.Criteria, inputs:List[Val])->List[types.GradingInput[Val]]:
    grading_inputs = map( lambda val: types.GradingInput[Val](criteria=criteria, input=val),
        inputs)
    return list(grading_inputs)
