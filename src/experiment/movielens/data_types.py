from typing import List
import pydantic
from decimal import Decimal


class Movie(pydantic.BaseModel):
    title: str = pydantic.Field(description="The title of the movie")
    year: int = pydantic.Field(description="The year of the movie")
    genres: List[str] = pydantic.Field(description="The genres of the movie")


class MovieRating(Movie):
    approval: float = pydantic.Field(
        description="The group approval rate of the movie from 0.0 - 1.0", ge=0.0, le=1.0)
    median_rating: float = pydantic.Field(
        description="The median rating of the movie from 0.5 - 5.0 in 0.5 increments", ge=0.5, le=5.0)


class FullMovieRating(MovieRating):
    loo_approval: float = pydantic.Field(
        description="The leave-one-out approval rate of the movie from 0.0 - 1.0", ge=0.0, le=1.0)
    global_approval: float = pydantic.Field(
        description="The global approval rate of the movie from 0.0 - 1.0", ge=0.0, le=1.0)
    avg_rating: float = pydantic.Field(
        description="The average rating of the movie from 1.0 - 5.0 in 0.5 increments", ge=1.0, le=5.0)
    std_deviation: float = pydantic.Field(
        description="The standard deviation of the movie's ratings", ge=0.0)
