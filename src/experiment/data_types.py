from datetime import datetime
from typing import List, Literal, Optional, TypeVar
from attr import dataclass
import seevals.data_types as types
import dspy
import pydantic
from collections.abc import Callable

T = TypeVar('T', bound=pydantic.BaseModel)
Z = TypeVar('Z', bound=pydantic.BaseModel)
type GetScoreFromDastasetFunction = Callable[[Z], int | float]
type GetEvaluationCriteria = Callable[[Z], types.Criteria]


class ExperimentConfig[T](pydantic.BaseModel):
    model: str = "openai/bedrock-sonnet-37"
    api_base: str = "http://localhost:4000"
    api_key: str = "noop"
    seed: int = 42
    metadata: T


class ExperimentInstance[T](pydantic.BaseModel):
    split_ratio: float = 0.8
    noise: float = 0.0
    seed: int = 42
    optimization: Literal["light", "medium", "heavy"] = "light"


@dataclass
class RunExperimentConfig[T]:
    lm: dspy.LM
    cfg: ExperimentInstance[T]
    get_criteria: GetEvaluationCriteria


@dataclass
class EvaluationDatasets[T]:
    eval: List[types.EvalData[T]]
    test_eval: List[types.EvalData[T]]


@dataclass
class DatasetSplit:
    train: List[dspy.Example]
    test: List[dspy.Example]


class ExperimentResult(pydantic.BaseModel):
    label: str
    instance: ExperimentInstance
    baseline_score: float
    optimized_score: float
    optimized_program_path: str
    timestamp: datetime


class ExperimentRunManifest(pydantic.BaseModel):
    run_id: str
    config: ExperimentConfig
    noise_params: List[float]
    created_at: datetime
    completed_at: Optional[datetime] = None
    labels: List[str]


class ExperimentRunResults(pydantic.BaseModel):
    run_id: str
    results: List[ExperimentResult]
