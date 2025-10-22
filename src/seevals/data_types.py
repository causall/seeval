import random
import pydantic
import json
import dspy
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from .path_utils import path_exists_in_model
import jsonpath_ng as jp


T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)
Z = TypeVar('Z', bound=pydantic.BaseModel)


class ForwardModule(Protocol[T, R]):
    def forward(self, params: T, /) -> dspy.Prediction:
        ...

    def __call__(self, params: T, /) -> dspy.Prediction:
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


class Sample(pydantic.BaseModel):
    num_samples: int


class View(pydantic.BaseModel):
    views: List[str]


class EvalItem(pydantic.BaseModel, Generic[T]):
    id: str
    sample: Optional[Sample]
    view: View
    data: Optional[T]
    score: Optional[float] = pydantic.Field(
        default=None, description="The score for the evaluation")


class EvalDatum(pydantic.BaseModel, Generic[T]):
    group_id: str
    items: List[EvalItem[T]]
    rubric: Rubric


class EvalData(pydantic.BaseModel, Generic[T]):
    data: List[EvalDatum[T]]
    raw_data: T


class EvalItemConfig(pydantic.BaseModel):
    sample: Optional[Sample] = pydantic.Field(
        default=None, description="The sample for the evaluation")
    view: View = pydantic.Field(
        default=None, description="The view for the evaluation")
    rubric: Rubric = pydantic.Field(
        default=None, description="The rubric for the evaluation")


class EvalConfig(pydantic.BaseModel):
    seed: int = pydantic.Field(
        default=42, description="The seed for the random number generator")
    config: Dict[str, EvalItemConfig] = pydantic.Field(
        default={}, description="The configuration for the evaluation")
    path_exists_cache: Dict[str, bool] = pydantic.Field(
        default={}, description="The cache for the path existence")
    class_type: Type[Z] = pydantic.Field(
        default=None, description="The class type for the evaluation")

    def __init__(self,  class_type: Type[Z] = None):
        super().__init__()
        self.class_type = class_type
        self.path_exists_cache = {}

    def apply(self, instances: List[Z], seed: int = 42) -> List[EvalData[Z]]:
        tracking_path = ""
        value = ""
        dataset = []
        count = 0
        for instance in instances[0:2]:
            data: EvalData[Z] = []
            try:
                instance_data = instance.model_dump()
                for path, cfg in self.config.items():
                    tracking_path = path
                    jsonpath_expr = jp.parse(path)
                    matches = jsonpath_expr.find(instance_data)
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected 1 match for path {path} but got {len(matches)}, {matches}")
                    items = []
                    value = matches[0].value
                    if cfg.sample is not None:
                        random.seed(seed)
                        values = matches[0].value
                        values = random.sample(
                            values, cfg.sample.num_samples)
                        for i, v in enumerate(values):
                            value = v
                            items.append(EvalItem(
                                id=f"{path}[{i}]",
                                sample=cfg.sample,
                                view=cfg.view,
                                data=v))
                    else:
                        items.append(EvalItem(
                            id=path,
                            sample=None,
                            view=cfg.view,
                            data=value))

                    data.append(EvalDatum(
                        group_id=f"{count}",
                        items=items,
                        rubric=cfg.rubric
                    ))
                count += 1
            except Exception as e:
                raise ValueError(
                    f"Error applying evaluation config: {e} for instance no:{count} value:{value} and path:{tracking_path}")
            dataset.append(EvalData(data=data, raw_data=instance_data))
        return dataset

    # has validation checking for the path against the class type

    def add(self, path: str, sample: Optional[Sample], view: View, rubric: Rubric) -> "EvalConfig":

        if not self.path_exists_cache.get(path, False) and not path_exists_in_model(self.class_type, path):
            raise ValueError(
                f"Path {path} not found in class {self.class_type.__name__}")

        self.path_exists_cache[path] = True
        for v in view.views:
            if not self.path_exists_cache.get(v, False) and not path_exists_in_model(self.class_type, v):
                raise ValueError(
                    f"View {v} not found in class {self.class_type.__name__}")

        for v in view.views:
            self.path_exists_cache[v] = True

        self.config[path] = EvalItemConfig(
            sample=sample,
            view=view,
            rubric=rubric
        )
        return self


class EvalDatasetBuilder(Generic[Z]):
    @classmethod
    def build(cls, instance: Type[Z]) -> EvalConfig:
        return EvalConfig(
            class_type=instance
        )


def gen_eval_object(config: EvalConfig, instance: Z):
    seed = config.seed
    random.seed(seed)
