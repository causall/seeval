import pydantic
import json
import dspy
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol


T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)


class ForwardModule(Protocol[T, R]):
    def forward(self, params: T, /) -> dspy.Prediction:
        ...

    def get_value(self, prediction: dspy.Prediction) -> R:
        ...


class ResponseData(pydantic.BaseModel, Generic[T]):
    data: List[Optional[T]]
    debug: List[Optional[dspy.Prediction]]
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class Rubric(pydantic.BaseModel):
    ge: float = pydantic.Field(
        default=0.0, description="The minimum score", ge=0.0)
    le: float = pydantic.Field(
        default=1.0, description="The maximum score", ge=0.0)
    desc: str = pydantic.Field(
        default="", description="The description of the metric")
    scale: Optional[str] = pydantic.Field(
        default=None, description="The scale of metric")


class Criteria(pydantic.BaseModel):
    rubrics: List[Rubric] = dspy.InputField(
        description="The rubrics used for grading the content")
    max_total_score: float = pydantic.Field(
        default=0.0, description="The sum of the rubric scores, but that must not exceed the maximum score")


class SampleCriteria(pydantic.BaseModel):
    num_samples: int = pydantic.Field(
        default=10, description="The number of samples to use for the criteria")
    confidence: float = pydantic.Field(
        default=0.95, description="The confidence level for the criteria")
    error_margin: float = pydantic.Field(
        default=0.0, description="The error margin for the criteria")


V = TypeVar('V')


class GradingArgs[V](TypedDict):
    criteria: Criteria
    input: V


class GradingInput[V](TypedDict):
    criteria: Criteria
    input: V


class TotalScore(pydantic.BaseModel):
    total_score: float = pydantic.Field(
        default=0.0, description="The total score between 0.0 and 1.0", ge=0.0, le=1.0, decimal_places=2)


class Score(pydantic.BaseModel):
    score: float = pydantic.Field(
        default=0.0, description="The score between 0.0 and 1.0", ge=0.0, le=1.0, decimal_places=2)
