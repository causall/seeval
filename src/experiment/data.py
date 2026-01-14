import random
from typing import List, Literal
import dspy
from experiment import agents as exp_agents
from experiment import data_types as exp_types
from experiment import utils as exp_utils
from seevals import agents, data_types as types
from seevals import utils
from seevals.execute import run_parallel


def generate_sentiment_headlines(lm: dspy.LM, num_headlines: int, seed: int):
    sentiment_headlines_positive = run_parallel(
        exp_agents.SentimentHeadlineGeneratorModule(),
        [
            exp_types.SentimentGenerationArgs(
                sentiments=["positive"], num_headlines=num_headlines
            )
        ],
        lm,
        1,
    )

    sentiment_headlines_negative = run_parallel(
        exp_agents.SentimentHeadlineGeneratorModule(),
        [
            exp_types.SentimentGenerationArgs(
                sentiments=["negative"], num_headlines=num_headlines
            )
        ],
        lm,
        1,
    )

    sentiment_headlines_neutral = run_parallel(
        exp_agents.SentimentHeadlineGeneratorModule(),
        [
            exp_types.SentimentGenerationArgs(
                sentiments=["neutral"], num_headlines=num_headlines
            )
        ],
        lm,
        1,
    )

    positive = utils.get_response_data_list(sentiment_headlines_positive)
    negative = utils.get_response_data_list(sentiment_headlines_negative)
    neutral = utils.get_response_data_list(sentiment_headlines_neutral)

    rng = random.Random(seed)
    sentiments = positive + negative + neutral
    rng.shuffle(sentiments)
    return sentiments


def generate_evaluation_dataset(
    lm: dspy.LM, num_headlines: int, seed: int
) -> List[types.EvalData[exp_types.SentimentHeadline]]:
    sentiments = generate_sentiment_headlines(lm, num_headlines, seed)
    eval = types.EvalDatasetBuilder.build(exp_types.SentimentHeadline)

    QualityRubric = types.Rubric(
        id=1,
        ge=0,
        le=1,
        desc="the quality of the headline",
        scale="0 is don't like, 1 is like, the scale is either 0 or 1",
    )
    eval.add("$.headline", None, types.View(
        views=["$.headline"]), QualityRubric)

    evaluation_dataset = eval.apply(sentiments, seed=seed)
    return evaluation_dataset


def apply_persona_evaluation_to_dataset(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    persona: exp_types.PersonaRanking,
    noise: float = 0.0,
    seed: int = 42,
) -> List[types.EvalData[exp_types.SentimentHeadline]]:
    ranked_dataset = utils.clone_dataset(evaluation_dataset)
    ranked_dataset = exp_utils.apply_persona_ranking(
        ranked_dataset, persona, noise, seed
    )
    return ranked_dataset


def generate_personas() -> exp_types.Personas:
    positive_persona = exp_types.PersonaRanking(
        positive=1, negative=0, neutral=0)
    negative_persona = exp_types.PersonaRanking(
        positive=0, negative=1, neutral=0)
    neutral_persona = exp_types.PersonaRanking(
        positive=0, negative=0, neutral=1)
    extreme_persona = exp_types.PersonaRanking(
        positive=1, negative=1, neutral=0)
    equal_persona = exp_types.PersonaRanking(positive=1, negative=1, neutral=1)

    return exp_types.Personas(
        positive=positive_persona,
        negative=negative_persona,
        neutral=neutral_persona,
        extreme=extreme_persona,
        equal=equal_persona,
    )


def get_experiment_data(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    criteria: types.Criteria,
    split_ratio: float = 0.8,
) -> exp_types.DatasetSplit:
    train_examples, test_examples = utils.make_train_test_split_from_eval_dataset(
        evaluation_dataset,
        criteria,
        split_ratio,
        input_filter=lambda d: {"headline": d["headline"]},
    )
    return exp_types.DatasetSplit(train=train_examples, test=test_examples)


def sentiment_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    frac=None,
    return_results=None,
) -> float:
    return 1 if gold.scores[0][0].score == pred.scores[0][0].score else 0


def build_gepa_grader(
    lm: dspy.LM, optimization: str = Literal["light", "medium", "heavy"]
) -> tuple[agents.GraderGenerationModule[exp_types.Headline], dspy.GEPA]:
    grading_module = agents.make_semantic_grader(exp_types.Headline)

    teleprompter = dspy.GEPA(
        auto="light",
        reflection_lm=lm,
        metric=sentiment_metric,
        num_threads=15,
        track_stats=True,
    )
    return grading_module, teleprompter
