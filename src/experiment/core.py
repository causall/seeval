from collections.abc import Callable
from itertools import product
from typing import List, Literal, Optional
from attr import Factory, dataclass
import dspy
import seevals.data_types as types
import experiment.utils as exp_utils
import experiment.data_types as exp_types
# import experiment.sentiment.data as exp_data


def get_evaluation_criteria() -> types.Criteria:
    QualityRubric = types.Rubric(id=1, ge=0, le=1, desc="the quality of the headline",
                                 scale="0 is don't like, 1 is like, the scale is either 0 or 1")
    return types.Criteria(rubrics=[QualityRubric])


def setup_experiment_lm(model: str, api_base: str, api_key: str):
    # 3. Set the callback to DSPy setting so it will be applied to program execution
    # exp_utils.AgentLoggingCallback()])
    dspy.configure(callbacks=[])  # exp_utils.AgentLoggingCallback()])

    lm = dspy.LM(
        model=model,
        # model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
        api_base=api_base,
        api_key=api_key,
    )

    return lm

# you might want to get and apply the score function


@dataclass
class AutomatedRubricScoring:
    rubric: Optional[types.Rubric] = None
    score: exp_types.GetScoreFromDastasetFunction = Factory(lambda: (lambda _: (
        _ for _ in ()).throw(NotImplementedError("score function required"))))


def apply_score_to_eval_data[T](eval_data: List[types.EvalData[T]], get_score: exp_types.GetScoreFromDastasetFunction):
    for item in eval_data:
        item.data[0].items[0].data
        item.data[0].items[0].score = get_score(item.raw_data)
    return eval_data


def apply_score_to_eval_data2[T](eval_data: List[types.EvalData[T]],  scoring: List[AutomatedRubricScoring]):
    scoring_map = {s.rubric.id: s.score for s in scoring}
    for item in eval_data:
        for datum in item.data:
            for item in datum.items:
                item.score = scoring_map[datum.rubric.id](item.data)


"""
def apply_automated_scoring_to_eval_data[T](eval_data: List[types.EvalData[T]], automated_scoring: List[AutomatedRubricScoring]):
    for item in eval_data:
"""


def create_eval_from_data2(data: List, scoring: List[AutomatedRubricScoring], seed: int, data_type: type) -> List[types.EvalData]:
    eval = types.EvalDatasetBuilder.build(data_type)
    for s in scoring:
        eval.add("$", None, types.View(views=["$"]), s.rubric)
    eval_data = eval.apply(data, seed=seed)
    apply_score_to_eval_data2(eval_data, scoring)
    return eval_data


def create_eval_from_data[T](data: List[T], get_score: exp_types.GetScoreFromDastasetFunction, seed: int) -> exp_types.EvaluationDatasets[T]:
    eval = types.EvalDatasetBuilder.build(T)
    eval.add("$", None, types.View(views=["$"]), types.Rubric(
        id=-1, desc="Automatically scored from dataset"))
    eval_data = eval.apply(data, seed=seed)
    return apply_score_to_eval_data(eval_data, get_score)
