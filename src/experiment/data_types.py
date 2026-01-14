from attr import dataclass
from datetime import datetime
import dspy
import pydantic
from typing import List, Literal, Optional


class PersonaRanking(pydantic.BaseModel):
    positive: int
    negative: int
    neutral: int


class ExperimentConfig(pydantic.BaseModel):
    model: str = "openai/bedrock-sonnet-37"
    api_base: str = "http://localhost:4000"
    api_key: str = "noop"
    seed: int = 42
    num_headlines: int = 100


class ExperimentInstance(pydantic.BaseModel):
    persona: PersonaRanking
    split_ratio: float = 0.8
    noise: float = 0.1
    seed: int = 42
    optimization: Literal["light", "medium", "heavy"] = "light"


class Personas(pydantic.BaseModel):
    positive: PersonaRanking
    negative: PersonaRanking
    neutral: PersonaRanking
    extreme: PersonaRanking
    equal: PersonaRanking


@dataclass
class DatasetSplit:
    """Kept as dataclass since dspy.Example isn't Pydantic-serializable"""
    train: List[dspy.Example]
    test: List[dspy.Example]


class SentimentGenerationArgs(pydantic.BaseModel):
    sentiments: List[Literal["positive", "negative", "neutral"]] = pydantic.Field(
        description="a list of sentiments to generate a list of headlines that are associated with the sentiment")
    num_headlines: int = pydantic.Field(
        description="the number of headlines to generate for each sentiment", gt=0, le=100)


class SentimentHeadline(pydantic.BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = pydantic.Field(
        description="the sentiment of the headline")
    headline: str = pydantic.Field(
        description="the headline that is associated with the sentiment")


class Headline(pydantic.BaseModel):
    headline: str = pydantic.Field(
        description="A news, magazine, or other type of headline")


class SentimentHeadlineOutput(pydantic.RootModel[List[SentimentHeadline]]):
    root: List[SentimentHeadline]


# Storage models

class ExperimentResult(pydantic.BaseModel):
    """Result of a single experiment instance run"""
    persona_name: str
    instance: ExperimentInstance
    baseline_score: float
    optimized_score: float
    optimized_program_path: str
    timestamp: datetime


class ExperimentRunManifest(pydantic.BaseModel):
    """Manifest for an entire experiment run"""
    run_id: str
    config: ExperimentConfig
    noise_params: List[float]
    created_at: datetime
    completed_at: Optional[datetime] = None
    persona_names: List[str]


class ExperimentRunResults(pydantic.BaseModel):
    """All results from an experiment run"""
    run_id: str
    results: List[ExperimentResult]
