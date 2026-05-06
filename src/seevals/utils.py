import pydantic
from typing import List, TypeVar, Optional
from . import data_types as types
import dspy
from typing import Callable


def instructions(**kwargs):
    def deco(fn):
        fn.__doc__ = fn.__doc__.format(**kwargs)
        return fn

    return deco


LT = TypeVar("LT", bound=pydantic.BaseModel)


def write_eval_dataset(path: str, eval_dataset: List[types.EvalData[LT]]):
    with open(path, "w") as f:
        for eval_data in eval_dataset:
            f.write(eval_data.model_dump_json() + "\n")


def load_eval_dataset(path: str) -> List[types.EvalData[LT]]:
    with open(path, "r") as f:
        return [types.EvalData.model_validate_json(line) for line in f]


def eval_data_to_examples(
    eval_dataset: List[types.EvalData[LT]],
    criteria: types.Criteria,
    include_outputs: bool = True,
    input_filter: Optional[Callable[[dict], dict]] = None,
) -> List[dspy.Example]:
    """Convert EvalData to dspy.Example.

    Args:
        input_filter: Optional function to filter/transform raw_data before passing to examples.
                      Useful for excluding fields the model shouldn't see (e.g., labels).
    """
    examples = []
    for eval_data in eval_dataset:
        input_data = eval_data.raw_data
        if input_filter:
            input_data = input_filter(input_data)

        example_kwargs = {"criteria": criteria, "input": input_data}

        if include_outputs and eval_data.data:
            scores = []
            for datum in eval_data.data:
                for item in datum.items:
                    if item.score is not None:
                        scores.append(
                            types.ScoredRubric(
                                rubric_id=datum.rubric.id,
                                score=item.score,
                                json_path=item.id,
                            )
                        )
            if scores:
                example_kwargs["scores"] = [scores]

        examples.append(dspy.Example(**example_kwargs).with_inputs("criteria", "input"))

    return examples


def make_train_test_split(
    examples: List[dspy.Example], split_ratio: float = 0.8
) -> types.Tuple[List[dspy.Example], List[dspy.Example]]:
    train_examples = examples[: int(len(examples) * split_ratio)]
    test_examples = examples[int(len(examples) * split_ratio) :]
    return train_examples, test_examples


def make_train_test_split_from_eval_dataset(
    eval_dataset: List[types.EvalData[LT]],
    criteria: types.Criteria,
    split_ratio: float = 0.8,
    input_filter: Optional[Callable[[dict], dict]] = None,
) -> types.Tuple[List[dspy.Example], List[dspy.Example]]:
    examples = eval_data_to_examples(eval_dataset, criteria, input_filter=input_filter)
    train_examples, test_examples = make_train_test_split(examples, split_ratio)
    return train_examples, test_examples


def get_response_data_list[T](
    response: types.ResponseData[pydantic.RootModel[List[T]]],
) -> List[T]:
    return response.data[0].root


def clone_dataset[T](dataset: List[types.EvalData[T]]) -> List[types.EvalData[T]]:
    return list(map(lambda x: x.model_copy(deep=True), dataset))
