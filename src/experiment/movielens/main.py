from pathlib import Path
import argparse
import random
import time
from typing import List, Literal
from attr import dataclass
from experiment.core import create_eval_from_data
from experiment.movielens.fetch_dataset import fetch_movielens
from experiment.movielens.group_dataset import SystematicSampleResult, filter_movies, get_ratings_matrix, load_datasets, SetupConfig, write_config_to_disk, load_config_from_disk
from experiment.movielens import group_dataset as gd, stats
from experiment.movielens.data_types import FullMovieRating, MovieRating
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def create_dataset_from_metrics(df: pd.DataFrame, data_type: Literal["train", "val", "test"]) -> MovieRating:
    def create_movie_rating_from_movie(sample: pd.DataFrame) -> FullMovieRating:
        avg_rating = sample['rating'].mean()
        std_deviation = sample['rating'].std()
        median_rating = sample['rating'].median()
        loo_approval_rate = sample['loo_approval'].mean()
        global_approval_rate = sample['global_approval'].mean()
        approval_rate = loo_approval_rate if data_type == "train" else global_approval_rate
        data_type = data_type
        return FullMovieRating(
            title=sample['title'].values[0],
            year=sample['year'].values[0],
            genres=sample['genres'].values[0],
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
    rng = random.Random(config.seed)
    sample_results = gd.load_sample_results_from_disk(Path(config.output_file))
    cache = establish_cache()

    def get_score(data: MovieRating) -> int:
        return data.rating

    movie_rating_metrics = create_movie_rating_metrics(cache.filtered_ratings,
                                                       sample_results[0:1], rng, config.exp_valid_movie_count)
    movie_rating_training_datasets = create_dataset_from_metrics(
        movie_rating_metrics[0].training_metrics, "train")
    movie_rating_validation_datasets = create_dataset_from_metrics(
        movie_rating_metrics[0].validation_metrics, "val")
    movie_rating_test_datasets = create_dataset_from_metrics(
        movie_rating_metrics[0].test_metrics, "test")

    import pdb
    pdb.set_trace()

    for movie_rating_dataset in movie_rating_datasets[0:1]:
        train_set = create_eval_from_data(
            movie_rating_dataset, get_score, config.seed)
        test_set = create_eval_from_data(
            movie_rating_dataset, get_score, config.seed)


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
