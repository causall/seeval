from typing import TypeVar, List
import dspy
from . import data_types as types
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
Val = TypeVar('Val')


def make_grading_inputs(criteria: types.Criteria, inputs: List[Val]) -> List[types.GradingInput[Val]]:
    grading_inputs = map(lambda val: types.GradingInput[Val](criteria=criteria, input=val),
                         inputs)
    return list(grading_inputs)


def make_grading_inputs_from_eval_dataset(criteria: types.Criteria, eval_dataset: List[types.EvalData[Val]]) -> List[types.GradingInput[Val]]:
    grading_inputs = map(lambda eval_data: types.GradingInput[Val](criteria=criteria, input=eval_data.raw_data),
                         eval_dataset)
    return list(grading_inputs)

def make_difference_feedback_metric(criteria: types.Criteria):
    rubric_titles = {r.id: r.title for r in criteria.rubrics}
    def difference_feedback_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, frac=None, return_results=None) -> ScoreWithFeedback:
        score = 0.0
        pred_map = {s.rubric_id: s.score for s in pred.scores[0]}
        feedbacks = []
        for s in gold.scores[0]:
            title = rubric_titles.get(s.rubric_id) or f"rubric id: {s.rubric_id}" 
            diff = s.score - pred_map[s.rubric_id]
            score = 1.0 - abs(diff)
            if diff < 0:
                feedbacks.append(f"The {title} should be lower. You were different by {diff:.2f}.")
            elif diff > 0:
                feedbacks.append(f"The {title} should be higher. You were different by {diff:.2f}.")
            else:
                feedbacks.append("")
        return ScoreWithFeedback(score=score,
                                 feedback=" ".join(feedbacks))

    return difference_feedback_metric 

