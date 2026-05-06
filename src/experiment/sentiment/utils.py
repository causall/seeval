# 1. Define a custom callback class that extends BaseCallback class

import math
import random
from typing import List
from dspy.utils.callback import BaseCallback
from seevals import data_types as types
from experiment.sentiment import data_types as exp_types


class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())


def apply_persona_ranking(evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]], persona: exp_types.PersonaRanking, noise: float = 0.0, seed: int = 42) -> List[types.EvalData[exp_types.SentimentHeadline]]:
    rng = random.Random(seed)
    for data in evaluation_dataset:
        score = 0
        match data.raw_data['sentiment']:
            case "positive":
                score = persona.positive
            case "negative":
                score = persona.negative
            case "neutral":
                score = persona.neutral
        # apply noise to score inverting it's natural direction
        if rng.random() < noise:
            score = abs(1-score)

        data.data[0].items[0].score = score

    return evaluation_dataset
