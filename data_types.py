import pydantic
import json
import dspy
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol


T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)

class ForwardModule(Protocol[T, R]):
    def forward(self, params: T, /)-> R:
        ...

class ResponseData(pydantic.BaseModel, Generic[T]):
    data: List[Optional[T]]

class Rubric(pydantic.BaseModel):
    ge: float = pydantic.Field(default=0.0, description="The minimum score", ge=0.0)
    le: float = pydantic.Field(default=1.0, description="The maximum score", ge=0.0)
    desc: str = pydantic.Field(default="", description="The description of the metric")

class Criteria(pydantic.BaseModel):
    rubrics: List[Rubric] = dspy.InputField(description="The rubrics used for grading the content")
    max_total_score: float = pydantic.Field(default=0.0, le=1.0, description="The sum of the rubric scores, but that must not exceed the maximum score")

V = TypeVar('V')
class GradingArgs[V](TypedDict):
    criteria: Criteria
    input: V

class GradingInput[V](TypedDict):
    criteria: Criteria
    input: V

class TotalScore(pydantic.BaseModel):
    total_score: float = pydantic.Field(default=0.0, description="The total score between 0.0 and 1.0",ge=0.0, le=1.0,decimal_places=2)

class Score(pydantic.BaseModel):
    score: float = pydantic.Field(default=0.0, description="The score between 0.0 and 1.0",ge=0.0, le=1.0,decimal_places=2)
