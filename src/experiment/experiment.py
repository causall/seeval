from itertools import product
from typing import List
import dspy
import seevals.data_types as types
import experiment.utils as exp_utils
import experiment.data_types as exp_types
import experiment.data as exp_data


def get_evaluation_criteria() -> types.Criteria:
    QualityRubric = types.Rubric(id=1, ge=0, le=1, desc="the quality of the headline",
                                 scale="0 is don't like, 1 is like, the scale is either 0 or 1")
    return types.Criteria(rubrics=[QualityRubric])


def setup_experiment_lm(model: str, api_base: str, api_key: str):
    # 3. Set the callback to DSPy setting so it will be applied to program execution
    dspy.configure(callbacks=[exp_utils.AgentLoggingCallback()])

    lm = dspy.LM(
        model=model,
        # model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
        api_base=api_base,
        api_key=api_key,
    )

    return lm


def run_experiment(lm: dspy.LM, evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
                   test_eval_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
                   cfg: exp_types.ExperimentInstance) -> (dspy.Module, dspy.GEPA):
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


def build_experiment_instances(evaluation_dataset, persona: exp_types.PersonaRanking, noises: [float], optimizations: [Literal["light", "medium", "heavy"]]):
    return [exp_types.ExperimentInstance(
        persona=p,
        split_ratio=0.8,
        noise=noise,
        seed=exp_types.ExperimentConfig().seed,
        optimization=optimization
    ) for p, noise, optimization in product([persona], noises, optimizations)]
