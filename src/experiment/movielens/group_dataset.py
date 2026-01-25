from typing import List
import pandas as pd
import numpy as np
from experiment.movielens.fetch_dataset import get_dataset_dir
from itertools import combinations


def load_datasets():
    movies = pd.read_csv(get_dataset_dir() / "ml-32m/movies.csv")
    regex = r"^(.*?)\s+\((\d{4})\)$"
    # Extract into two new columns
    movies[['title', 'year']] = movies['title'].str.extract(regex)
    movies['genres_split'] = movies['genres'].apply(split_genre)
    movies['genre_list'] = movies['genres'].apply(
        split_and_expand_genre)
    ratings = pd.read_csv(get_dataset_dir() / "ml-32m/ratings.csv")
    return movies, ratings


def filter_movies(movies: pd.DataFrame, ratings: pd.DataFrame, num_reviews: int, num_individual_user_reviews: int) -> pd.DataFrame:
    movie_counts = ratings.groupby('movieId').size()

    filtered_min_reviews = movie_counts[movie_counts >= num_reviews].index
    import pdb

    df_filtered_ratings = ratings[ratings['movieId'].isin(
        filtered_min_reviews)]

    user_counts = df_filtered_ratings.groupby('userId').size()
    filtered_min_users = user_counts[user_counts >=
                                     num_individual_user_reviews].index

    df_filtered_ratings = df_filtered_ratings[df_filtered_ratings['userId'].isin(
        filtered_min_users)]

    return df_filtered_ratings


def expand_movies_with_genres_columns(movies: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    genre_cols = pd.DataFrame({
        genre: movies['genre_list'].apply(
            lambda x, g=genre: 1 if g in x else 0)
        for genre in genres
    }, index=movies.index)
    return pd.concat([movies, genre_cols], axis=1)


def build_genres(movies: pd.DataFrame) -> pd.DataFrame:
    movies['genres_split'] = movies['genres'].apply(split_genre)
    set_of_genres = set()
    for genre in movies['genres_split']:
        set_of_genres.update(genre)

    genres = list(combinations(set_of_genres, 2))
    genres = [f"{g[0]}_{g[1]}" for g in genres] + list(set_of_genres)

    return genres


def build_genre_ratings(ratings: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    # Binary presence: 1 if genre in list, 0 otherwise
    genre_presence = pd.DataFrame({
        genre: ratings['genre_list'].apply(
            lambda x, g=genre: 1 if g in x else 0)
        for genre in genres
    }, index=ratings.index)

    import pdb
    pdb.set_trace()
    # Multiply each column by the rating
    genre_cols = genre_presence.mul(ratings['rating'], axis=0)

    return pd.concat([ratings, genre_cols], axis=1)

# get the ratings per user per sample to calculate the preference matrix
# for the set of users where we'll weighted average over all that users
# genre preferences
# we will take the average user genre rating over all the overall average of individual ratings
# and this will compute the difference between them and that is our preference weight
# the outcome of any test should then reflect the overall preference of the group
# so there is a difference that can happen between the training set / validation set / and test set


def get_random_sample_of_users(ratings: pd.DataFrame, num_users: int, rng: np.random.Generator) -> pd.DataFrame:
    unique_users = ratings['userId'].unique()
    sampled_users = rng.choice(unique_users, size=num_users, replace=False)
    ratings_sample = ratings[ratings['userId'].isin(sampled_users)]
    return ratings_sample


# assumed it has the genre broken down ratings into columns
def get_user_preference_matrix(expanded_ratings: pd.DataFrame) -> pd.DataFrame:
    # 1. Group by userId and calculate the mean for all columns
    new_df = expanded_ratings.groupby('userId').mean()

    # 2. Calculate the overall average for each user
    new_df['user_overall_avg'] = new_df.mean(axis=1)

    # 2. Calculate the global average for each column in the original (or grouped) data
    global_averages = new_df.mean()

    # 3. Create the "difference from average" statistics
    # This subtracts the global mean from each user's mean
    diff_df = new_df.subtract(new_df['user_overall_avg'], axis=0).drop(
        columns=['user_overall_avg'])

    # 4. (Optional) Rename columns to distinguish the new statistics
    diff_df = diff_df.add_suffix('delta_user_avg')

    # 5. Combine them into one final DataFrame
    final_df = pd.concat([new_df, diff_df], axis=1)

    return final_df


def split_and_expand_genre(genre: str) -> List[str]:
    split_genres = split_genre(genre)
    pairs = list(combinations(split_genres, 2))
    full_genres = [f"{p[0]}_{p[1]}" for p in pairs] + split_genres
    return full_genres


def split_genre(genre: str) -> List[str]:
    return [g.strip() for g in genre.split("|")]
