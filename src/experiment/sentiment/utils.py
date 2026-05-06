import random
from typing import List

from seevals import data_types as types
from experiment.sentiment import data_types as exp_types


def apply_persona_ranking(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    persona: exp_types.PersonaRanking,
    noise: float = 0.0,
    seed: int = 42,
) -> List[types.EvalData[exp_types.SentimentHeadline]]:
    rng = random.Random(seed)
    for data in evaluation_dataset:
        score = 0
        match data.raw_data["sentiment"]:
            case "positive":
                score = persona.positive
            case "negative":
                score = persona.negative
            case "neutral":
                score = persona.neutral
        if rng.random() < noise:
            score = abs(1 - score)

        data.data[0].items[0].score = score

    return evaluation_dataset
