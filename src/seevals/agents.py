import dspy
import pydantic
from dataclasses import dataclass
from pydantic import Field
from typing import Sequence, Annotated, Unpack, List, Tuple, TypedDict, Type, TypeVar, Generic, Callable
from dspy import InputField, OutputField
import numpy as np
from . import data_types as types


class QA(pydantic.BaseModel):
    question: str = pydantic.Field(
        description="The question to ask",
        # min_length=50,
        # max_length=200
    )
    answer: str = pydantic.Field(
        description="The answer to the question",
        # min_length=50,
        # max_length=200
    )
    suggestion: str = pydantic.Field(
        description="A suggested answer to the question. The suggested answer should be grounded in some evidence when possible",
        # min_length=50,
        # max_length=200
    )
    suggestion_explanation: str = pydantic.Field(
        description="Explanation of the suggestion, why the suggested answer is chosen",
        # min_length=50,
        # max_length=200
    )
    model_config = pydantic.ConfigDict(
        extra='forbid')  # Disallow extra fields


class QABaseModel(pydantic.BaseModel):
    interview: List[QA] = pydantic.Field(
        default_factory=list,
        description="Questions and they're associated answers in chronological order",
        min_length=5,
        max_length=10
    )


class MetaAnalysisEntity(pydantic.BaseModel):
    name: str = pydantic.Field(
        description="The unique name of the entity")
    type: str = pydantic.Field(
        description="The type of the entity")


class MetaAnalysisIndices(pydantic.BaseModel):
    subsections_index: List[str] = pydantic.Field(
        description="A unique list of subsections")
    entities_index: List[MetaAnalysisEntity] = pydantic.Field(
        description="A unique list of entities in associated subsections")
    relationships_index: List[Tuple[MetaAnalysisEntity, MetaAnalysisEntity]] = pydantic.Field(
        description="A unique tuple of entities describing a relationship between the entities")


class MetaAnalysis(pydantic.BaseModel):
    analysis_overview: str = pydantic.Field(
        description="The analysis overview of the subsections, entities and relationships")
    analysis_indices: MetaAnalysisIndices = pydantic.Field(
        description="The indices of the analysis to enable efficient querying")


class InterviewAnalysis(dspy.Signature):
    """
    You are a tokenomics, economics, and behavioral psychology expert. You are skilled at genrating causal loop diagrams and mapping relationships.
    You are greate at planning how analysis should be conducted, including identifying key variables, data sources, and considerations. Then condensing them into
    a well formatted analysis plan. The plan should be developed utilizing the interview to guide the analysis.

    You reply with the full updated feedback not only the updated information. You respond with a complete response, unless a conintuation is specified.
    """
    scenario: str = InputField(
        decription="The tokenomic or economic scenario to analyze")
    interview = InputField(
        desc=f"An interview questions and answers to use to complete the analysis, in this format {QABaseModel.model_json_schema()}")
    history = InputField(
        description="The history of the analysis so far")
    continuation = InputField(
        description="The portion of the output that needs to be completed")

    analysis_plan: MetaAnalysis = OutputField(description='The analysis plan')
    # analysis_plan = generate_json_output_field(MetaAnalysis)


class AnalysisPlanningResult(MetaAnalysis):
    ...


class AnalysisPlanning(dspy.Signature):
    scenario = InputField(
        decription="The tokenomic or economic scenario to analyze")
    analysis_plan: MetaAnalysis = OutputField(description='The analysis plan')


class ScenarioArgs(pydantic.BaseModel):
    scenario: str


GRADER_PROMPT_1 = """
System:
  You are an expert medical grader. Compare the **Reference Answer** to the **Model's Answer** and produce **only** a JSON object with:
    • **result**: a float between 0.0 and 1.0
    • **steps**: a list of reasoning steps (each with a `"description"` and a `"conclusion"`)

  Scoring rubric (start at 0.0, then add or subtract):
    1. Exact lexical match: **+0.15**
    2. Clinical synonym (e.g. “withdrawal of thought” ↔ “thought withdrawal”): **+0.35**
    3. Same disease family (e.g. two viral encephalitides): **+0.35**
    4. Partial term overlap (e.g. “ulcer” in both phrases): **+0.15**
    5. Completely unrelated: **-0.10**

  • If multiple criteria apply, sum their weights (max 1.0).
  • Cap the final score to the [0.0, 1.0] range.
  • In your **steps**, show which rule you applied and the running subtotal.
"""


def make_ordered_score_tuple(
    rubrics: list[types.Rubric]
) -> Type[tuple]:

    # 1) Build each Annotated float type
    types = tuple(
        Annotated[str, rubric]
        for rubric in rubrics
    )
    # 2) Dynamically subscribe Tuple[...] to that tuple of types
    return type(types)


class SemanticSignature[V](dspy.Signature):
    criteria: types.Criteria = InputField(
        description="The criteria for grading")
    input: V = InputField(description="The input to be graded")
    score: float = OutputField(
        description="The score of how the input meets the criteria")


class ContrastiveSignature[V, O](dspy.Signature):
    """
    Your goal is to modify the input to fail the criteria, and return the modified input as output.
    How much you fail the criteria will be determined by the noise factor specified. Use the noise factor to
    adjust the input to impact the score of how the input meets the criteria by that amount.
    """
    criteria: types.Criteria = InputField(
        description="The criteria for grading")
    noise_factor: float = InputField(
        description="The noise factor for modifying the input")
    input: V = InputField(description="The input to be graded")
    output: O = OutputField(
        description="The modified input, to meet the contrastive criteria")


T = TypeVar('T')


class GradingInput[T](TypedDict):
    criteria: types.Criteria
    input: T


class ContrastiveInput[T](TypedDict):
    criteria: types.Criteria
    noise_factor: float
    input: T


C = TypeVar('C')


class GradingResult(pydantic.BaseModel):
    score: float


I = TypeVar('I', bound='pydantic.BaseModel')


class GraderGenerationModule(dspy.Module, Generic[I]):
    def __init__(self, input_type: Type[I]):
        self.grader = dspy.ChainOfThought(
            SemanticSignature[GradingInput[input_type]])

    def forward(self, input: GradingInput[I]) -> dspy.Prediction:
        return self.grader(**input)

    def get_value(self, prediction: dspy.Prediction) -> GradingResult:
        return prediction.score


def make_semantic_grader(InputType: Type[I]) -> GraderGenerationModule[I]:
    return GraderGenerationModule(InputType)


class GraderContrastiveModule(dspy.Module, Generic[I]):
    def __init__(self, input_type: Type[I]):
        self.contrast = dspy.ChainOfThought(
            ContrastiveSignature[ContrastiveInput[input_type], input_type])

    def forward(self, input: ContrastiveInput[I]) -> dspy.Prediction:
        return self.contrast(**input)

    def get_value(self, prediction: dspy.Prediction) -> I:
        return prediction.output


# Normal distribution noise factor application to inputs
In = TypeVar('In')


def from_grading_inputs(inputs: Sequence[GradingInput[In]], mean_noise_factor: float, rng: np.random.Generator = np.random.default_rng(42)) -> List[ContrastiveInput[In]]:
    noise_factors = rng.normal(loc=mean_noise_factor, size=len(inputs))
    return [from_grading_input(input, noise_factor) for input, noise_factor in zip(inputs, noise_factors)]


In2 = TypeVar('In2')


def from_grading_input(input: GradingInput[In2], noise_factor: float) -> ContrastiveInput[In2]:
    return ContrastiveInput(criteria=input['criteria'], input=input['input'], noise_factor=noise_factor)


def make_contrastive_grader(InputType: Type[I]) -> GraderContrastiveModule[I]:
    return GraderContrastiveModule(InputType)


class InterviewGenerationModule(dspy.Module):
    def __init__(self):
        self.analysis_plan = dspy.ChainOfThought(AnalysisPlanning)

    def forward(self, scenario: ScenarioArgs) -> dspy.Prediction:
        return self.analysis_plan(**scenario.model_dump())

    def get_value(self, prediction: dspy.Prediction) -> AnalysisPlanningResult:
        return prediction.analysis_plan

# I want to generate synthetic data that produces a range of scores based on the criteria
# is this what I want to do? To produce a distribution of good and bad examples
