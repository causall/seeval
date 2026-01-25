from experiment.movielens.fetch_dataset import fetch_movielens
from experiment.movielens.group_dataset import filter_movies, load_datasets
from experiment.movielens import group_dataset as gd
import pandas as pd


def main():
    fetch_movielens()
    movies, ratings = load_datasets()
    filtered_ratings = filter_movies(movies, ratings, num_reviews=30,
                                     num_individual_user_reviews=30)
    genre_list = gd.build_genres(movies)
    expanded_movies = gd.expand_movies_with_genres_columns(
        movies, genre_list)
    expanded_set = pd.merge(filtered_ratings, expanded_movies, on='movieId')
    import pdb
    pdb.set_trace()
    result = gd.build_genre_ratings(expanded_set, genre_list)
    user_preference_matrix = gd.get_user_preference_matrix(result)
    import pdb
    pdb.set_trace()
    print(ratings.head())
    print(movies.head())


if __name__ == "__main__":
    main()
