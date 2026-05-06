from attr import dataclass
import dspy
import pydantic
from typing import List, Literal

import seevals.data_types as types


class PersonaRanking(pydantic.BaseModel):
    positive: int
    negative: int
    neutral: int


class Personas(pydantic.BaseModel):
    positive: PersonaRanking
    negative: PersonaRanking
    neutral: PersonaRanking
    extreme: PersonaRanking
    equal: PersonaRanking


class SentimentMetadata(pydantic.BaseModel):
    num_headlines: int = 100


class SentimentGenerationArgs(pydantic.BaseModel):
    sentiments: List[Literal["positive", "negative", "neutral"]] = pydantic.Field(
        description="a list of sentiments to generate a list of headlines that are associated with the sentiment"
    )
    num_headlines: int = pydantic.Field(
        description="the number of headlines to generate for each sentiment",
        gt=0,
        le=300,
    )


class SentimentHeadline(pydantic.BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = pydantic.Field(
        description="the sentiment of the headline"
    )
    headline: str = pydantic.Field(
        description="the headline that is associated with the sentiment"
    )


class Headline(pydantic.BaseModel):
    headline: str = pydantic.Field(
        description="A news, magazine, or other type of headline"
    )


class SentimentHeadlineOutput(pydantic.RootModel[List[SentimentHeadline]]):
    root: List[SentimentHeadline]


@dataclass
class SplitDatasets:
    """Train / val / test slices of the persona-scored evaluation dataset."""

    train: List[types.EvalData[SentimentHeadline]]
    validation: List[types.EvalData[SentimentHeadline]]
    test: List[types.EvalData[SentimentHeadline]]


@dataclass
class SplitExamples:
    """Same splits, materialized as scored, input-filtered dspy.Examples."""

    train: List[dspy.Example]
    validation: List[dspy.Example]
    test: List[dspy.Example]
