from typing import List
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from experiment.movielens.fetch_dataset import get_dataset_dir
from itertools import combinations

import pydantic
from pydantic import Field

pd.set_option("display.max_columns", None)


class SystematicSampleResult(pydantic.BaseModel):
    movie_ids: List[int] = Field(description="The list of movie ids in the sample")
    user_ids: List[int] = Field(description="The list of user ids in the sample")
    valid_movie_count: int = Field(
        description="The number of valid movies in the sample"
    )


class SetupConfig(pydantic.BaseModel):
    """Configuration for dataset generation setup"""

    seed: int = Field(default=43, description="Random seed for reproducibility")
    num_runs: int = Field(default=10000, description="Number of sampling iterations")
    valid_movie_count: int = Field(
        default=50,
        description="Minimum valid movies threshold for filtering samples for data collection",
    )
    num_users: int = Field(default=5, description="Number of users per sample")
    output_file: str = Field(description="Output JSONL filename")
    exp_valid_movie_count: int = Field(
        description="The number of valid movies required to run an experiment",
        default=100,
    )


def write_config_to_disk(config: SetupConfig, path: Path):
    """Write setup config to JSON file"""
    path.write_text(config.model_dump_json(indent=2))


def load_config_from_disk(path: Path) -> SetupConfig:
    """Load setup config from JSON file"""
    return SetupConfig.model_validate_json(path.read_text())


def write_sample_results_to_disk(
    sample_results: List[SystematicSampleResult], path: Path
):
    # Create an adapter for the list type

    # --- Writing to Disk ---
    sample_results_json = [result.model_dump_json() for result in sample_results]
    path.write_text("\n".join(sample_results_json))


def load_sample_results_from_disk(path: Path) -> List[SystematicSampleResult]:
    # Create an adapter for the list type
    sample_results = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            sample_result = SystematicSampleResult.model_validate_json(line)
            sample_results.append(sample_result)
    return sample_results


def load_datasets():
    movies = pd.read_csv(get_dataset_dir() / "ml-32m/movies.csv")
    regex = r"^(.*?)\s+\((\d{4})\)$"
    # Extract into two new columns
    movies[["title", "year"]] = movies["title"].str.extract(regex)
    movies["genres_split"] = movies["genres"].apply(split_genre)
    movies["genre_list"] = movies["genres"].apply(split_and_expand_genre)
    ratings = pd.read_csv(get_dataset_dir() / "ml-32m/ratings.csv")
    genre_list = build_genres(movies)
    return movies, ratings, genre_list


def filter_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    num_reviews: int,
    num_individual_user_reviews: int,
) -> pd.DataFrame:
    movie_counts = ratings.groupby("movieId").size()

    filtered_min_reviews = movie_counts[movie_counts >= num_reviews].index

    df_filtered_ratings = ratings[ratings["movieId"].isin(filtered_min_reviews)]

    user_counts = df_filtered_ratings.groupby("userId").size()
    filtered_min_users = user_counts[user_counts >= num_individual_user_reviews].index

    df_filtered_ratings = df_filtered_ratings[
        df_filtered_ratings["userId"].isin(filtered_min_users)
    ]

    return df_filtered_ratings


def expand_movies_with_genres_columns(
    movies: pd.DataFrame, genres: List[str]
) -> pd.DataFrame:
    genre_cols = pd.DataFrame(
        {
            genre: movies["genre_list"].apply(lambda x, g=genre: 1 if g in x else 0)
            for genre in genres
        },
        index=movies.index,
    )
    return pd.concat([movies, genre_cols], axis=1)


def build_genres(movies: pd.DataFrame) -> pd.DataFrame:
    movies["genres_split"] = movies["genres"].apply(split_genre)
    no_genres_listed = "(no genres listed)"
    set_of_genres = set()
    for genre in movies["genres_split"]:
        if no_genres_listed not in genre:
            set_of_genres.update(genre)

    genres = list(combinations(set_of_genres, 2))
    genres = (
        [f"{g[0]}_{g[1]}" for g in genres] + list(set_of_genres) + [no_genres_listed]
    )

    return genres


def build_genre_ratings_old(ratings: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # Binary presence: 1 if genre in list, 0 otherwise
    genre_presence = pd.DataFrame(
        {
            genre: ratings["genre_list"].apply(lambda x, g=genre: 1 if g in x else 0)
            for genre in genres
        },
        index=ratings.index,
    )

    import pdb

    pdb.set_trace()
    # Multiply each column by the rating
    genre_cols = genre_presence.mul(ratings["rating"], axis=0)

    return pd.concat([ratings, genre_cols], axis=1)


def build_genre_ratings(ratings: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # Convert genre_list to sets for O(1) lookup
    genre_sets = ratings["genre_list"].apply(frozenset).tolist()

    # Build presence matrix with numpy - much faster than per-column apply
    presence = np.array([[1 if g in gs else 0 for g in genres] for gs in genre_sets])

    # Multiply by rating column (broadcasting)
    genre_cols = presence * ratings["rating"].values[:, np.newaxis]

    # Create DataFrame with genre columns
    genre_df = pd.DataFrame(genre_cols, columns=genres, index=ratings.index)

    return pd.concat([ratings, genre_df], axis=1)


# get the ratings per user per sample to calculate the preference matrix
# for the set of users where we'll weighted average over all that users
# genre preferences
# we will take the average user genre rating over all the overall average of individual ratings
# and this will compute the difference between them and that is our preference weight
# the outcome of any test should then reflect the overall preference of the group
# so there is a difference that can happen between the training set / validation set / and test set

# so


def create_movie_and_user_idices(ratings: pd.DataFrame) -> tuple[dict, dict]:
    user_movie_idx = ratings.groupby("userId")["movieId"].agg(frozenset).to_dict()
    movie_user_idx = ratings.groupby("movieId")["userId"].agg(frozenset).to_dict()
    return user_movie_idx, movie_user_idx


def get_indices_cache_path(
    dataset_name: str, num_reviews: int, num_individual_user_reviews: int
) -> Path:
    return (
        get_dataset_dir()
        / f".cache/indices_{dataset_name}_minrev{num_reviews}_minind{num_individual_user_reviews}.pkl"
    )


def create_movie_and_user_indices_cached(
    ratings: pd.DataFrame,
    dataset_name: str = "ml-32m",
    num_reviews: int = 0,
    num_individual_user_reviews: int = 0,
    force_recompute: bool = False,
) -> tuple[dict, dict]:
    cache_path = get_indices_cache_path(
        dataset_name, num_reviews, num_individual_user_reviews
    )

    if not force_recompute and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    user_movie_idx, movie_user_idx = create_movie_and_user_idices(ratings)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((user_movie_idx, movie_user_idx), f)

    return user_movie_idx, movie_user_idx


def get_systematic_sample(
    ratings: pd.DataFrame,
    num_users: int,
    user_movie_idx: dict,
    movie_user_idx: dict,
    rng: np.random.Generator,
) -> SystematicSampleResult:
    """
    1. Pick random users
    2. Find movies where >= min_users_per_movie of them overlap
    3. Return stats about what's available
    """
    # Sample users
    unique_users = list(user_movie_idx.keys())
    sampled_users = set(rng.choice(unique_users, size=num_users, replace=False))
    #  print(f"sampled_users size: {len(sampled_users)}")
    #  print(f"sampled_users type: {type(list(sampled_users)[0])}")

    # Check one movie's users
    # sample_mid, sample_users = next(iter(movie_user_idx.items()))
    #  print(f"movie_user_idx user type: {type(list(sample_users)[0])}")
    #  print(f"intersection size: {len(sample_users & sampled_users)}")

    # Count overlap per movie
    movie_overlap = {
        mid: len(users & sampled_users) for mid, users in movie_user_idx.items()
    }

    # Bucket by overlap count
    from collections import Counter

    overlap_distribution = Counter(movie_overlap.values())

    # Filter movies meeting threshold
    valid_movies = [mid for mid, cnt in movie_overlap.items() if cnt >= num_users]

    stats = {
        "overlap_distribution": dict(sorted(overlap_distribution.items())),
        "movies_by_threshold": {
            k: sum(1 for c in movie_overlap.values() if c >= k)
            for k in range(1, num_users + 1)
        },
        "valid_movie_count": len(valid_movies),
        "movie_ids": valid_movies,
        "user_ids": list(sampled_users),
    }

    return SystematicSampleResult(**stats)
    # return ratings[ratings['movieId'].isin(valid_movies) & ratings['userId'].isin(sampled_users)], stats


def get_random_sample_of_users_by_movie(
    ratings: pd.DataFrame, num_users: int, num_movies: int, rng: np.random.Generator
) -> pd.DataFrame:
    unique_users = ratings["userId"].unique()
    sampled_users = rng.choice(unique_users, size=num_users, replace=False)
    unique_movies = ratings[ratings["userId"].isin(sampled_users)]["movieId"].unique()
    sampled_movies = rng.choice(unique_movies, size=num_movies, replace=False)
    ratings_sample = ratings[ratings["movieId"].isin(sampled_movies)]
    return ratings_sample


def get_random_sample_of_users(
    ratings: pd.DataFrame, num_users: int, rng: np.random.Generator
) -> pd.DataFrame:
    unique_users = ratings["userId"].unique()
    sampled_users = rng.choice(unique_users, size=num_users, replace=False)
    ratings_sample = ratings[ratings["userId"].isin(sampled_users)]
    return ratings_sample


def get_ratings_matrix(
    ratings: pd.DataFrame, movie_ids: List[int], user_ids: List[int]
) -> pd.DataFrame:
    return ratings[
        ratings["movieId"].isin(movie_ids) & ratings["userId"].isin(user_ids)
    ]


# This is almost right but I think I might need to weight
# the ratings by number of ratings per genre, like one movie you
# rate a 5.0 should not score higher than 50 you rated 4.5


def get_user_preference_matrix(
    expanded_ratings: pd.DataFrame, genres: List[str]
) -> pd.DataFrame:
    # Use np.nan instead of pd.NA - significantly faster
    data = expanded_ratings[["userId"] + genres].copy()
    data[genres] = data[genres].replace(0, np.nan)

    # Groupby is already optimized, hard to beat
    grouped = data.groupby("userId")[genres].mean()

    # NumPy for the rest - avoid pandas overhead
    values = grouped.values
    user_avg = np.nanmean(values, axis=1, keepdims=True)
    diff_values = values - user_avg

    # Single DataFrame construction at the end
    diff_cols = [f"{g}delta_user_avg" for g in genres]

    return pd.DataFrame(
        np.hstack([values, user_avg, diff_values]),
        index=grouped.index,
        columns=genres + ["user_overall_avg"] + diff_cols,
    )


def split_and_expand_genre(genre: str) -> List[str]:
    split_genres = split_genre(genre)
    pairs = list(combinations(split_genres, 2))
    full_genres = [f"{p[0]}_{p[1]}" for p in pairs] + split_genres
    return full_genres


def split_genre(genre: str) -> List[str]:
    return [g.strip() for g in genre.split("|")]
