import pydantic
import json
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol

T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)

class ForwardModule(Protocol[T, R]):
    def forward(self, params: T, /)-> R:
        ...

class ResponseData(pydantic.BaseModel, Generic[T]):
    data: List[Optional[T]]
