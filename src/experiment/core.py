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


"""
def get_experiment_data(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    criteria: types.Criteria,
    split_ratio: float = 0.8,
    input_filter: Optional[Callable[[dict], dict]] = None
) -> exp_types.DatasetSplit:
    train_examples, test_examples = exp_utils.make_train_test_split_from_eval_dataset(
        evaluation_dataset,
        criteria,
        split_ratio,
        input_filter=input_filter,
    )
    return exp_types.DatasetSplit(train=train_examples, test=test_examples)
"""

"""
def run_experiment[T](datasets: exp_types.EvaluationDatasets[T],
                      cfg: exp_types.RunExperimentConfig[T]) -> tuple[dspy.Module, dspy.GEPA, float, float]:

    dataset = exp_data.apply_persona_evaluation_to_dataset(
        evaluation_dataset, cfg.persona, cfg.noise, cfg.seed)
    test_dataset = exp_data.apply_persona_evaluation_to_dataset(
        test_eval_dataset, cfg.persona, cfg.noise, cfg.seed)

    dataset_split = exp_data.get_experiment_data(
        dataset, get_evaluation_criteria(), cfg.split_ratio)
    test_dataset_split = exp_data.get_experiment_data(
        test_dataset, get_evaluation_criteria(), 0.0)

    baseline_evaluate = dspy.Evaluate(devset=test_dataset_split.test, metric=exp_data.sentiment_metric,
                                      num_threads=10, display_progress=True, display_table=0, max_errors=999)
    (grader, teleprompter) = exp_data.build_gepa_grader(lm, cfg.optimization)
    baseline_score = baseline_evaluate(grader)
    optimized_program = teleprompter.compile(
        grader,
        trainset=dataset_split.train,
        valset=dataset_split.test,
    )

    optimized_score = baseline_evaluate(optimized_program)
    return optimized_program, teleprompter, baseline_score, optimized_score
"""

"""
def build_experiment_instances(evaluation_dataset, persona: exp_types.PersonaRanking, noises: [float], optimizations: [Literal["light", "medium", "heavy"]]):
    return [exp_types.ExperimentInstance(
        persona=p,
        split_ratio=0.8,
        noise=noise,
        seed=exp_types.ExperimentConfig().seed,
        optimization=optimization
    ) for p, noise, optimization in product([persona], noises, optimizations)]
"""
