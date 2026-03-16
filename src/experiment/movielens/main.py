from pathlib import Path
import argparse
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
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

import seevals.data_types as types

pd.options.mode.copy_on_write = "warn"


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


def run_experiment(config: SetupConfig):
    """Run the experiment with the given config"""
    experiment_config = ExperimentConfig(
        model="openai/qwen3-235b", metadata="movie_rating", seed=config.seed)
    # setup the experiment lm
    lm = setup_experiment_lm(experiment_config.model,
                             experiment_config.api_base, experiment_config.api_key)
    dspy.configure(lm=lm)
    rng = random.Random(config.seed)
    sample_results = gd.load_sample_results_from_disk(Path(config.output_file))
    cache = establish_cache()

    movie_rating_metrics = create_movie_rating_metrics(cache.filtered_ratings,
                                                       sample_results[-3:-2], rng, config.exp_valid_movie_count)

    def add_metadata(metrics: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
        return metrics.merge(movies[['movieId', 'title', 'year', 'genres']], on='movieId', how='left')

    movie_rating_training_datasets = create_dataset_from_metrics(
        add_metadata(movie_rating_metrics[0].training_metrics, cache.movies), "train")
    movie_rating_validation_datasets = create_dataset_from_metrics(
        add_metadata(movie_rating_metrics[0].validation_metrics, cache.movies), "val")
    movie_rating_test_datasets = create_dataset_from_metrics(
        add_metadata(movie_rating_metrics[0].test_metrics, cache.movies), "test")

    def get_median_score(data: FullMovieRating) -> int:
        return data['median_rating']

    def get_avg_score(data: FullMovieRating) -> int:
        return data['avg_rating']

    def get_global_approval_score(data: FullMovieRating) -> int:
        return data['global_approval']

    def get_loo_approval_score(data: FullMovieRating) -> int:
        return data['loo_approval']

    training_automated_scoring = [
        AutomatedRubricScoring(rubric=types.Rubric(
            id=1, desc="Median Rating"), score=get_median_score),
        AutomatedRubricScoring(rubric=types.Rubric(
            id=2, desc="LOO Approval"), score=get_loo_approval_score),
    ]
    test_validation_automated_scoring = [
        AutomatedRubricScoring(rubric=types.Rubric(
            id=1, desc="Median Rating"), score=get_median_score),
        AutomatedRubricScoring(rubric=types.Rubric(
            id=2, desc="global Approval"), score=get_global_approval_score),
    ]

    criteria = types.Criteria(rubrics=[types.Rubric(id=1, desc="The median rating of movie cohort id: 2788", scale="0.5 is the lowest rating, 5.0 is the highest rating", ge=0.5, le=5.0),
                                       types.Rubric(id=2, desc="The percentage of movie cohort id: 2788 that like this significantly more than other movies", scale="0.0 - 1.0", ge=0.0, le=1.0)])

    def input_filter(data: FullMovieRating) -> dict:
        return {"title": data["title"], "year": data["year"], "genres": data["genres"]}

    train_set = create_eval_from_data2(
        movie_rating_training_datasets, training_automated_scoring, config.seed, FullMovieRating)
    validation_set = create_eval_from_data2(
        movie_rating_validation_datasets, test_validation_automated_scoring, config.seed, FullMovieRating)
    test_set = create_eval_from_data2(
        movie_rating_test_datasets, test_validation_automated_scoring, config.seed, FullMovieRating)

    validation_examples, _ = make_train_test_split_from_eval_dataset(
        validation_set, criteria, 1, input_filter)
    train_examples, _ = make_train_test_split_from_eval_dataset(
        train_set, criteria, 1, input_filter)
    test_examples, _ = make_train_test_split_from_eval_dataset(
        test_set, criteria, 1, input_filter)

    grading_module = agents.make_semantic_grader(
        FullMovieRating, "You are able to guess at the preference of cohort id: 2788")

    def movie_rating_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, frac=None, return_results=None) -> ScoreWithFeedback:
        approval_id = criteria.rubrics[1].id
        median_rating_id = criteria.rubrics[0].id
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

        return ScoreWithFeedback(score=score, feedback=f"{feedback} {median_rating_feedback}")
        # .ScoreWithFeedback(score=score, feedback=feedback)

    time_start = time.time()
    teleprompter = dspy.GEPA(
        auto="light",
        reflection_lm=lm,
        metric=movie_rating_metric,
        num_threads=30,
        track_stats=True,
    )

    baseline_evaluate = dspy.Evaluate(devset=test_examples, metric=movie_rating_metric,
                                      num_threads=10, display_progress=True, display_table=0, max_errors=999)

    baseline_score = baseline_evaluate(grading_module)
    optimized_program = teleprompter.compile(
        grading_module,
        trainset=train_examples,
        valset=validation_examples,
    )

    optimized_score = baseline_evaluate(optimized_program)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    import pdb
    pdb.set_trace()

    return optimized_program, teleprompter, baseline_score, optimized_score


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
