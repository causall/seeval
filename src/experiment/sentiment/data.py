import random
from typing import List

import dspy

from seevals import data_types as types
from seevals import utils
from seevals.execute import run_parallel

from experiment.sentiment import agents as exp_agents
from experiment.sentiment import data_types as exp_types
from experiment.sentiment import utils as exp_utils


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
        title="quality",
        ge=0,
        le=1,
        desc="the quality of the headline",
        scale="0 is don't like, 1 is like, the scale is either 0 or 1",
    )
    eval.add("$.headline", None, types.View(views=["$.headline"]), QualityRubric)

    return eval.apply(sentiments, seed=seed)


def apply_persona_evaluation_to_dataset(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    persona: exp_types.PersonaRanking,
    noise: float = 0.0,
    seed: int = 42,
) -> List[types.EvalData[exp_types.SentimentHeadline]]:
    ranked_dataset = utils.clone_dataset(evaluation_dataset)
    return exp_utils.apply_persona_ranking(ranked_dataset, persona, noise, seed)


def generate_personas() -> exp_types.Personas:
    return exp_types.Personas(
        positive=exp_types.PersonaRanking(positive=1, negative=0, neutral=0),
        negative=exp_types.PersonaRanking(positive=0, negative=1, neutral=0),
        neutral=exp_types.PersonaRanking(positive=0, negative=0, neutral=1),
        extreme=exp_types.PersonaRanking(positive=1, negative=1, neutral=0),
        equal=exp_types.PersonaRanking(positive=1, negative=1, neutral=1),
    )
