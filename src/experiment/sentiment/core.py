"""Sentiment experiment stages mirroring the movielens layout.

`build_split_datasets` -> `build_split_examples` -> `run_gepa_experiment` is
the same three-stage pipeline `experiment.movielens.main` uses; we share the
LM setup, callback, and difference-feedback metric across experiments.
"""

import random
import time
from itertools import product
from typing import List, Literal

import dspy

import seevals.data_types as types
from seevals import agents
from seevals.agent_util import make_difference_feedback_metric
from seevals.utils import (
    clone_dataset,
    make_train_test_split_from_eval_dataset,
)

from experiment.core import setup_experiment_lm  # re-exported for sentiment.py
from experiment.data_types import ExperimentInstance
from experiment.sentiment import data_types as exp_types
from experiment.sentiment import utils as exp_utils

__all__ = [
    "setup_experiment_lm",
    "get_evaluation_criteria",
    "build_split_datasets",
    "build_split_examples",
    "run_gepa_experiment",
    "build_experiment_instances",
]


def get_evaluation_criteria() -> types.Criteria:
    """Single binary rubric. `title` feeds nicer feedback strings into GEPA."""
    return types.Criteria(
        rubrics=[
            types.Rubric(
                id=1,
                title="quality",
                ge=0,
                le=1,
                desc="the quality of the headline",
                scale="0 is don't like, 1 is like, the scale is either 0 or 1",
            )
        ]
    )


def _movie_headline_input_filter(data) -> dict:
    return {"headline": data["headline"]}


def build_split_datasets(
    evaluation_dataset: List[types.EvalData[exp_types.SentimentHeadline]],
    persona: exp_types.PersonaRanking,
    instance: ExperimentInstance,
    train_test_split: float = 0.8,
    train_val_split: float = 0.8,
) -> exp_types.SplitDatasets:
    """Apply persona+noise scoring and slice into train/val/test EvalData lists.

    Mirrors movielens's `build_split_datasets`: the master dataset is cloned,
    scored against the persona (with noise injection seeded by `instance.seed`),
    then a deterministic 80/20 train/test split is taken, and a 80/20
    train/val split is taken inside the train half.
    """
    scored = clone_dataset(evaluation_dataset)
    scored = exp_utils.apply_persona_ranking(
        scored, persona, instance.noise, instance.seed
    )

    rng = random.Random(instance.seed)
    rng.shuffle(scored)

    cut_test = int(len(scored) * train_test_split)
    train_pool = scored[:cut_test]
    test = scored[cut_test:]

    cut_val = int(len(train_pool) * train_val_split)
    train = train_pool[:cut_val]
    validation = train_pool[cut_val:]

    return exp_types.SplitDatasets(train=train, validation=validation, test=test)


def build_split_examples(
    splits: exp_types.SplitDatasets,
    criteria: types.Criteria,
    seed: int,
) -> exp_types.SplitExamples:
    """Convert each split's EvalData -> dspy.Example list (no further splitting).

    `make_train_test_split_from_eval_dataset(..., split_ratio=1.0)` returns
    `(all_examples, [])`, the same trick movielens uses to reuse the helper
    for whole-dataset materialization while keeping scores attached.
    """
    del seed  # split is already done; helper takes a split_ratio not a seed.

    train, _ = make_train_test_split_from_eval_dataset(
        splits.train, criteria, 1.0, _movie_headline_input_filter
    )
    validation, _ = make_train_test_split_from_eval_dataset(
        splits.validation, criteria, 1.0, _movie_headline_input_filter
    )
    test, _ = make_train_test_split_from_eval_dataset(
        splits.test, criteria, 1.0, _movie_headline_input_filter
    )
    return exp_types.SplitExamples(train=train, validation=validation, test=test)


def run_gepa_experiment(
    lm: dspy.LM,
    examples: exp_types.SplitExamples,
    criteria: types.Criteria,
    optimization: Literal["light", "medium", "heavy"] = "light",
) -> tuple[dspy.Module, dspy.GEPA, float, float]:
    """Baseline + GEPA optimization on a prepared example split.

    Uses `make_difference_feedback_metric` from seevals so GEPA gets textual
    feedback per example. For binary {0,1} rubrics the score is identical to
    the previous exact-match metric (`1 - |diff|` collapses to 0/1).
    """
    grading_module = agents.make_semantic_grader(exp_types.Headline)
    metric = make_difference_feedback_metric(criteria)

    teleprompter = dspy.GEPA(
        auto=optimization,
        reflection_lm=lm,
        metric=metric,
        num_threads=15,
        track_stats=True,
    )

    baseline_evaluate = dspy.Evaluate(
        devset=examples.test,
        metric=metric,
        num_threads=10,
        display_progress=True,
        display_table=0,
        max_errors=999,
    )

    t0 = time.time()
    baseline_score = baseline_evaluate(grading_module)
    optimized_program = teleprompter.compile(
        grading_module,
        trainset=examples.train,
        valset=examples.validation,
    )
    optimized_score = baseline_evaluate(optimized_program)
    print(f"GEPA stage took {time.time() - t0:.1f}s")

    return optimized_program, teleprompter, baseline_score, optimized_score


def build_experiment_instances(
    noises: List[float],
    optimizations: List[Literal["light", "medium", "heavy"]],
    seed: int = 42,
    split_ratio: float = 0.8,
) -> List[ExperimentInstance]:
    """Cartesian product of noise * optimization. Persona is tracked separately
    in the main loop so we don't shoehorn it onto the generic ExperimentInstance."""
    return [
        ExperimentInstance(
            split_ratio=split_ratio,
            noise=noise,
            seed=seed,
            optimization=optimization,
        )
        for noise, optimization in product(noises, optimizations)
    ]
