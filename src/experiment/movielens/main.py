from pathlib import Path
import argparse
import hashlib
import logging
import random
import time
from typing import List, Literal
from attr import dataclass
import dspy
from experiment.core import AutomatedRubricScoring, create_eval_from_data, create_eval_from_data2, setup_experiment_lm
from experiment.movielens.fetch_dataset import fetch_movielens
from experiment.movielens.group_dataset import SystematicSampleResult, filter_movies, get_ratings_matrix, load_datasets, SetupConfig, write_config_to_disk, load_config_from_disk
from experiment.movielens import group_dataset as gd, stats
from experiment.movielens.data_types import FullMovieRating, MovieRating
from experiment.data_types import ExperimentConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seevals import agents
from seevals.execute import run_parallel
from seevals.utils import make_train_test_split_from_eval_dataset
from seevals.agent_util import make_difference_feedback_metric
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

import seevals.data_types as types

pd.options.mode.copy_on_write = "warn"

# Silence GEPA teleprompter chatter (per-iteration pareto/score logs).
# Flip to INFO to re-enable.
logging.getLogger("dspy.teleprompt.gepa.gepa").setLevel(logging.WARNING)

# Silence the per-minibatch "Average Metric: X / Y (Z%)" tqdm bars that GEPA
# emits via bootstrap_trace_data (which hardcodes display_progress=True).
# Targeted monkeypatch: force display_progress=False only for that code path.
from dspy.teleprompt import bootstrap_finetune as _gepa_bf  # noqa: E402
_orig_bf_evaluate = _gepa_bf.Evaluate


def _quiet_bf_evaluate(*args, **kwargs):
    kwargs["display_progress"] = False
    return _orig_bf_evaluate(*args, **kwargs)


_gepa_bf.Evaluate = _quiet_bf_evaluate
logging.getLogger("dspy.evaluate.evaluate").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MovieLens experiment dataset generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--setup", metavar="FILENAME", type=str,
        help="Generate dataset and save to FILENAME (should be .jsonl)")
    parser.add_argument(
        "--num-users", type=int, default=5,
        help="Number of users per sample")
    parser.add_argument(
        "--num-runs", type=int, default=10000,
        help="Number of sampling iterations")
    parser.add_argument(
        "--valid-movie-count", type=int, default=50,
        help="Minimum valid movie count threshold for filtering samples")
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--analysis", metavar="CONFIG_FILE", type=str,
        help="Run analysis using the specified config JSON file")
    return parser.parse_args()


def get_config_path(output_path: Path) -> Path:
    """Derive config path from output path: foo.jsonl -> foo_config.json"""
    return output_path.with_suffix("").parent / (output_path.stem + "_config.json")


@dataclass
class Cache:
    user_movie_idx: dict
    movie_user_idx: dict
    filtered_ratings: pd.DataFrame
    movies: pd.DataFrame


def establish_cache() -> Cache:
    """Establish cache for the dataset"""
    fetch_movielens()
    movies, ratings, genre_list = load_datasets()
    filtered_ratings = filter_movies(
        movies, ratings, num_reviews=1000, num_individual_user_reviews=200)
    (user_movie_idx, movie_user_idx) = gd.create_movie_and_user_indices_cached(
        filtered_ratings, num_individual_user_reviews=200, num_reviews=1000)
    return Cache(user_movie_idx, movie_user_idx, filtered_ratings, movies)


def run_setup(config: SetupConfig):
    """Run dataset generation with the given config"""
    fetch_movielens()
    movies, ratings, genre_list = load_datasets()
    rng = np.random.default_rng(seed=config.seed)
    cache = establish_cache()
    const_sets = []
    time_start = time.time()
    for _ in range(config.num_runs):
        sampled_ratings = gd.get_systematic_sample(
            cache.filtered_ratings, config.num_users, cache.user_movie_idx, cache.movie_user_idx, rng)
        if sampled_ratings.valid_movie_count > config.valid_movie_count:
            const_sets.append(sampled_ratings)
    time_end = time.time()

    print(f"Time taken: {time_end - time_start} seconds")
    print(
        f"Generated {len(const_sets)} samples with valid_movie_count > {config.valid_movie_count}")

    const_sets.sort(key=lambda x: x.valid_movie_count, reverse=True)

    output_path = Path(config.output_file)
    config_path = get_config_path(output_path)

    gd.write_sample_results_to_disk(const_sets, output_path)
    write_config_to_disk(config, config_path)

    print(f"Wrote samples to: {output_path}")
    print(f"Wrote config to: {config_path}")


@dataclass
class TrainTestMovieIds:
    train_movie_ids: List[int]
    test_movie_ids: List[int]
    split_ratio: float


@dataclass
class TrainValTestMetrics:
    training_metrics: pd.DataFrame
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame


def split_calculation_movie_ids(rng: random.Random, movie_ids: List[int], split_ratio: float = 0.8) -> TrainTestMovieIds:
    train_movie_ids = []
    test_movie_ids = []
    rng.shuffle(movie_ids)
    train_movie_ids = movie_ids[:int(len(movie_ids)*split_ratio)]
    test_movie_ids = movie_ids[int(len(movie_ids)*split_ratio):]
    return TrainTestMovieIds(train_movie_ids, test_movie_ids, split_ratio)


def split_train_test_validation_movie_ids(rng: random.Random, movie_ids: List[int], split_ratio: float = 0.8) -> TrainTestMovieIds:
    train_movie_ids = []
    test_movie_ids = []
    validation_movie_ids = []
    rng.shuffle(movie_ids)
    train_movie_ids = movie_ids[:int(len(movie_ids)*split_ratio)]
    test_movie_ids = movie_ids[int(len(movie_ids)*split_ratio):]
    return TrainTestMovieIds(train_movie_ids, test_movie_ids, split_ratio)


def split_train_test_movie_ids(rng: random.Random, movie_ids: List[int], split_ratio: float = 0.8) -> TrainTestMovieIds:
    train_movie_ids = []
    test_movie_ids = []
    rng.shuffle(movie_ids)
    train_movie_ids = movie_ids[:int(
        len(movie_ids)*split_ratio)]
    test_movie_ids = movie_ids[int(
        len(movie_ids)*split_ratio):]

    return TrainTestMovieIds(train_movie_ids, test_movie_ids, split_ratio)


def run_analysis(config: SetupConfig):
    """Run dataset analysis with the given config"""
    rng = random.Random(config.seed)
    sample_results = gd.load_sample_results_from_disk(Path(config.output_file))
    cache = establish_cache()
    samp = sample_results[0]
    sample = cache.filtered_ratings[cache.filtered_ratings['movieId'].isin(
        samp.movie_ids) & cache.filtered_ratings['userId'].isin(samp.user_ids)]
    merged_sample = sample.merge(cache.movies, on='movieId', how='left')
    result = stats.calc_leave_one_out_metrics(merged_sample)
    movie_approval_rate = result.groupby('movieId')['loo_approval'].mean()
    movie_approval_rate.hist(bins=10)

    plt.show()
    import pdb
    pdb.set_trace()
    print(sample)


def create_dataset_from_metrics(df: pd.DataFrame, data_type: Literal["train", "val", "test"]) -> List[FullMovieRating]:
    def create_movie_rating_from_movie(sample: pd.DataFrame) -> FullMovieRating:
        avg_rating = sample['rating'].mean()
        std_deviation = sample['rating'].std()
        median_rating = sample['rating'].median()
        # check for if loo and global exists in the first place if not set to 0
        loo_approval_rate = sample['loo_approval'].mean(
        ) if 'loo_approval' in sample else 0.0
        global_approval_rate = sample['global_approval'].mean(
        ) if 'global_approval' in sample else 0.0
        approval_rate = loo_approval_rate if data_type == "train" else global_approval_rate

        return FullMovieRating(
            title=sample['title'].values[0],
            year=sample['year'].values[0],
            genres=sample['genres'].values[0].split('|'),
            approval=approval_rate,
            median_rating=median_rating,
            avg_rating=avg_rating,
            std_deviation=std_deviation,
            loo_approval=loo_approval_rate,
            global_approval=global_approval_rate
        )

    return [
        create_movie_rating_from_movie(group)
        for _, group in df.groupby("movieId")
    ]

# write using the global metrics from the training data to calculate the metrics on the test data
# so we accurately tag the values that we'd try and predict with validation data


def calculate_metrics_on_training_data(cache: Cache, train_test_set: TrainTestMovieIds, user_ids: List[int]) -> List[MovieRating]:
    train_movie_ratings = cache.filtered_ratings[cache.filtered_ratings['movieId'].isin(
        train_test_set.train_movie_ids) & cache.filtered_ratings['userId'].isin(user_ids)]
    merged_train_movie_ratings = train_movie_ratings.merge(
        cache.movies, on='movieId', how='left')
    result = stats.calc_leave_one_out_metrics(merged_train_movie_ratings)
    stats.calc_metrics(cache, train_test_set, user_ids)
    return result


def calculate_metrics_on_test_data(cache: Cache, train_test_set: TrainTestMovieIds, user_ids: List[int]) -> List[MovieRating]:
    test_movie_ratings = cache.filtered_ratings[cache.filtered_ratings['movieId'].isin(
        train_test_set.test_movie_ids) & cache.filtered_ratings['userId'].isin(user_ids)]


def create_movie_rating_metrics(data: pd.DataFrame, sample_results: List[SystematicSampleResult], rng: random.Random, minimum_size: int = 100) -> List[MovieRating]:
    datasets = []
    scale = 1.0
    for sample_result in sample_results:
        if sample_result.valid_movie_count >= minimum_size:
            # construct test set and training + validation set
            train_test_movie_ids = split_train_test_movie_ids(rng,
                                                              sample_result.movie_ids, 0.8)
            # construct training and validation matrix set
            training_movie_ids = split_train_test_movie_ids(rng,
                                                            train_test_movie_ids.train_movie_ids, 0.8)

            training_matrix = get_ratings_matrix(
                data, training_movie_ids.train_movie_ids, sample_result.user_ids)
            # calc training statistics
            training_metrics = stats.calc_leave_one_out_metrics(
                training_matrix, scale)

            validation_matrix = get_ratings_matrix(
                data, training_movie_ids.test_movie_ids, sample_result.user_ids)
            # calc validation metrics
            validation_metrics = stats.calc_test_statistics(
                training_metrics, validation_matrix, scale)

            test_matrix = get_ratings_matrix(
                data, train_test_movie_ids.test_movie_ids, sample_result.user_ids)
            # calc test metrics
            test_metrics = stats.calc_test_statistics(
                training_metrics, test_matrix, scale)
            # we need to split thetraining set again to be able to calculate the metrics on only training data to not leak
            # calculate the metrics on only training data to not leak
            # calculate the test set metrics against full training set data excluding the other test set metrics

            datasets.append(
                TrainValTestMetrics(training_metrics, validation_metrics, test_metrics))
    return datasets


def make_cohort_token(seed: int, cohort_idx: int) -> str:
    """Deterministic opaque anchor used to pin the grader to a single persona.

    The literal value has no semantic meaning; rotating it per cohort prevents
    a GEPA-optimized prompt from baking in cohort-specific phrasing that would
    transfer poorly. Same (seed, cohort_idx) always yields the same token, so
    produce-side and run-side can agree without passing state.
    """
    h = hashlib.blake2b(
        f"{seed}:{cohort_idx}".encode(), digest_size=3).hexdigest()
    return f"c-{h}"


def get_movie_rating_criteria(cohort_token: str = "c-placeholder") -> types.Criteria:
    """Evaluation criteria for the movielens experiment.

    `cohort_token` is an opaque anchor (see `make_cohort_token`) inlined into
    rubric descriptions to keep the grader committed to one consistent persona
    rather than averaging across "all users".
    """
    return types.Criteria(rubrics=[
        types.Rubric(
            title="median rating",
            id=1,
            desc=f"The median rating of movie cohort id: {cohort_token}",
            scale="0.5 is the lowest rating, 5.0 is the highest rating",
            ge=0.5, le=5.0,
        ),
        types.Rubric(
            title="approval rate",
            id=2,
            desc=f"The percentage of movie cohort id: {cohort_token} that like this significantly more than other movies",
            scale="0.0 - 1.0",
            ge=0.0, le=1.0,
        ),
    ])


def _movie_rating_input_filter(data) -> dict:
    return {"title": data["title"], "year": data["year"], "genres": data["genres"]}


def _get_median_score(data) -> float:
    return data['median_rating']


def _get_global_approval_score(data) -> float:
    return data['global_approval']


def _get_loo_approval_score(data) -> float:
    return data['loo_approval']


def _training_scoring() -> List[AutomatedRubricScoring]:
    return [
        AutomatedRubricScoring(rubric=types.Rubric(id=1, desc="Median Rating"),
                               score=_get_median_score),
        AutomatedRubricScoring(rubric=types.Rubric(id=2, desc="LOO Approval"),
                               score=_get_loo_approval_score),
    ]


def _test_validation_scoring() -> List[AutomatedRubricScoring]:
    return [
        AutomatedRubricScoring(rubric=types.Rubric(id=1, desc="Median Rating"),
                               score=_get_median_score),
        AutomatedRubricScoring(rubric=types.Rubric(id=2, desc="global Approval"),
                               score=_get_global_approval_score),
    ]


@dataclass
class SplitDatasets:
    train: List[FullMovieRating]
    validation: List[FullMovieRating]
    test: List[FullMovieRating]


def build_split_datasets(cache: Cache, sample_result, rng: random.Random,
                         exp_valid_movie_count: int,
                         total_examples: "int | None" = None) -> SplitDatasets:
    """Build FullMovieRating lists for train/val/test splits for a single cohort.

    If ``total_examples`` is set, the cohort's ``movie_ids`` are deterministically
    downsampled to exactly that size *before* the 80/20 -> 80/20 split, so every
    produced cohort yields identical train/val/test sizes.
    Returns None if the sample has fewer valid movies than
    ``max(exp_valid_movie_count, total_examples or 0)``.
    """
    required = max(exp_valid_movie_count, total_examples or 0)
    if sample_result.valid_movie_count < required:
        return None

    if total_examples is not None and len(sample_result.movie_ids) > total_examples:
        chosen = rng.sample(sample_result.movie_ids, total_examples)
        sample_result = sample_result.model_copy(update={
            "movie_ids": chosen,
            "valid_movie_count": total_examples,
        })

    threshold = total_examples if total_examples is not None else exp_valid_movie_count
    metrics_list = create_movie_rating_metrics(
        cache.filtered_ratings, [sample_result], rng, threshold)
    if not metrics_list:
        return None

    metrics = metrics_list[0]

    def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(
            cache.movies[['movieId', 'title', 'year', 'genres']],
            on='movieId', how='left')

    train = create_dataset_from_metrics(add_metadata(metrics.training_metrics), "train")
    validation = create_dataset_from_metrics(add_metadata(metrics.validation_metrics), "val")
    test = create_dataset_from_metrics(add_metadata(metrics.test_metrics), "test")
    return SplitDatasets(train=train, validation=validation, test=test)


@dataclass
class SplitExamples:
    train: List[dspy.Example]
    validation: List[dspy.Example]
    test: List[dspy.Example]


def build_split_examples(splits: SplitDatasets, criteria: types.Criteria,
                         seed: int) -> SplitExamples:
    """Convert FullMovieRating splits into scored, input-filtered dspy.Examples."""
    train_set = create_eval_from_data2(
        splits.train, _training_scoring(), seed, FullMovieRating)
    validation_set = create_eval_from_data2(
        splits.validation, _test_validation_scoring(), seed, FullMovieRating)
    test_set = create_eval_from_data2(
        splits.test, _test_validation_scoring(), seed, FullMovieRating)

    train_examples, _ = make_train_test_split_from_eval_dataset(
        train_set, criteria, 1, _movie_rating_input_filter)
    validation_examples, _ = make_train_test_split_from_eval_dataset(
        validation_set, criteria, 1, _movie_rating_input_filter)
    test_examples, _ = make_train_test_split_from_eval_dataset(
        test_set, criteria, 1, _movie_rating_input_filter)
    return SplitExamples(train=train_examples,
                         validation=validation_examples,
                         test=test_examples)


def make_movie_rating_metric(criteria: types.Criteria):
    """Build the GEPA metric closure for the given criteria."""
    approval_id = criteria.rubrics[1].id
    median_rating_id = criteria.rubrics[0].id

    def movie_rating_metric(gold: dspy.Example, pred: dspy.Prediction,
                            trace=None, frac=None, return_results=None) -> ScoreWithFeedback:
        score = 0.0
        feedback = ""
        median_rating_feedback = ""
        pred_map = {s.rubric_id: s for s in pred.scores[0]}
        for s in gold.scores[0]:
            if s.rubric_id == approval_id and approval_id in pred_map:
                diff = s.score - pred_map[approval_id].score
                score = 1.0 - abs(diff)
                if diff < 0:
                    feedback = f"The approval rate should be lower. You were different by {diff:.2f}."
                elif diff > 0:
                    feedback = f"The approval rate should be higher. You were different by {diff:.2f}."
                else:
                    feedback = ""
            if s.rubric_id == median_rating_id and median_rating_id in pred_map:
                diff = s.score - pred_map[median_rating_id].score
                if diff < 0:
                    median_rating_feedback = f"The median rating should be higher. You were different by {diff:.2f}."
                elif diff > 0:
                    median_rating_feedback = f"The median rating should be lower. You were different by {diff:.2f}."
                else:
                    median_rating_feedback = ""

        return ScoreWithFeedback(score=score,
                                 feedback=f"{feedback} {median_rating_feedback}")

    return movie_rating_metric


def run_gepa_experiment(lm, examples: SplitExamples, criteria: types.Criteria,
                        cohort_token: str):
    """Run baseline + GEPA optimization on a prepared example split.

    `cohort_token` must match the token baked into `criteria` so the grader
    instruction and rubric descriptions reference the same anchor.
    """
    grading_module = agents.make_semantic_grader(
        FullMovieRating)
        #f"You are able to guess at the preference of cohort id: {cohort_token}")
    metric = make_difference_feedback_metric(criteria)

    time_start = time.time()
    teleprompter = dspy.GEPA(
        auto="light",
        reflection_lm=lm,
        metric=metric,
        num_threads=30,
        track_stats=True,
    )

    baseline_evaluate = dspy.Evaluate(
        devset=examples.test, metric=metric, num_threads=10,
        display_progress=False, display_table=0, max_errors=999)

    baseline_score = baseline_evaluate(grading_module)
    optimized_program = teleprompter.compile(
        grading_module,
        trainset=examples.train,
        valset=examples.validation,
    )

    optimized_score = baseline_evaluate(optimized_program)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    return optimized_program, teleprompter, baseline_score, optimized_score


def run_experiment(config: SetupConfig):
    """Run the experiment with the given config"""
    experiment_config = ExperimentConfig(
        model="openai/qwen3-235b", metadata="movie_rating", seed=config.seed)
    lm = setup_experiment_lm(experiment_config.model,
                             experiment_config.api_base, experiment_config.api_key)
    dspy.configure(lm=lm)
    rng = random.Random(config.seed)
    sample_results = gd.load_sample_results_from_disk(Path(config.output_file))
    cache = establish_cache()

    cohort_idx = len(sample_results) - 3
    splits = build_split_datasets(
        cache, sample_results[cohort_idx], rng, config.exp_valid_movie_count)
    if splits is None:
        raise RuntimeError(
            "Selected sample_result does not meet exp_valid_movie_count threshold")

    cohort_token = make_cohort_token(config.seed, cohort_idx)
    criteria = get_movie_rating_criteria(cohort_token)
    examples = build_split_examples(splits, criteria, config.seed)
    return run_gepa_experiment(lm, examples, criteria, cohort_token)


"""
    samp = sample_results[0]
    sample = cache.filtered_ratings[cache.filtered_ratings['movieId'].isin(
        samp.movie_ids) & cache.filtered_ratings['userId'].isin(samp.user_ids)]
    merged_sample = sample.merge(cache.movies, on='movieId', how='left')
"""


def main():
    args = parse_args()

    if args.setup:
        config = SetupConfig(
            seed=args.seed,
            num_runs=args.num_runs,
            valid_movie_count=args.valid_movie_count,
            num_users=args.num_users,
            output_file=args.setup
        )
        run_setup(config)
    elif args.analysis:
        config = load_config_from_disk(Path(args.analysis))
        # run_analysis(config)
        run_experiment(config)
    else:
        print("No action specified. Use --setup <filename> to generate a dataset.")
        print("Use --analysis <config_file> to run analysis.")
        print("Run with --help for more options.")


if __name__ == "__main__":
    main()
