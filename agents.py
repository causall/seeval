import dspy
import pydantic
from dataclasses import dataclass
from pydantic import Field
from typing import Annotated, Unpack,List, Tuple, TypedDict, Type, TypeVar
from dspy import InputField, OutputField



class QA(pydantic.BaseModel):
    question: str = pydantic.Field(
        description="The question to ask",
        #min_length=50,
        #max_length=200
    )
    answer: str = pydantic.Field(
        description="The answer to the question",
        #min_length=50,
        #max_length=200
    )
    suggestion: str = pydantic.Field(
        description="A suggested answer to the question. The suggested answer should be grounded in some evidence when possible",
        #min_length=50,
        #max_length=200
    )
    suggestion_explanation: str = pydantic.Field(
        description="Explanation of the suggestion, why the suggested answer is chosen",
        #min_length=50,
        #max_length=200
    )
    model_config = pydantic.ConfigDict(
        extra='forbid')  # Disallow extra fields

class QABaseModel(pydantic.BaseModel):
    interview: List[QA] = pydantic.Field(
        default_factory=list,
        description="Questions and they're associated answers in chronological order",
        min_length = 5,
        max_length = 10
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
    scenario:str = InputField(
        decription="The tokenomic or economic scenario to analyze")
    interview = InputField(desc=f"An interview questions and answers to use to complete the analysis, in this format {QABaseModel.model_json_schema()}")
    history = InputField(
        description="The history of the analysis so far")
    continuation = InputField(description="The portion of the output that needs to be completed")

    analysis_plan: MetaAnalysis = OutputField(description='The analysis plan')
    # analysis_plan = generate_json_output_field(MetaAnalysis)

class AnalysisPlanningResult(pydantic.BaseModel):
    analysis_plan: MetaAnalysis

class AnalysisPlanning(dspy.Signature):
    scenario = InputField(
        decription="The tokenomic or economic scenario to analyze")
    analysis_plan: MetaAnalysis = OutputField(description='The analysis plan')

class ScenarioArgs(TypedDict):
    scenario: str


V = TypeVar('V')
class GradingArgs[V](TypedDict):
    criteria: List[str]
    input: V

class TotalScore(pydantic.BaseModel):
    total_score: float = pydantic.Field(default=0.0, description="The total score between 0.0 and 1.0",ge=0.0, le=1.0,decimal_places=2)

def make_rubric()
class Score(pydantic.BaseModel):
    score: float = pydantic.Field(default=0.0, description="The score between 0.0 and 1.0",ge=0.0, le=1.0,decimal_places=2)

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

class Rubric(pydantic.BaseModel):
    ge: float = pydantic.Field(default=0.0, description="The minimum score", ge=0.0)
    le: float = pydantic.Field(default=1.0, description="The maximum score", ge=0.0)
    desc: str = pydantic.Field(default="", description="The description of the metric")

def make_ordered_score_tuple(
    rubrics: list[Rubric]
) -> Type[tuple]:

    # 1) Build each Annotated float type
    types = tuple(
        Annotated[str, rubric]
        for rubric in rubrics
    )
    # 2) Dynamically subscribe Tuple[...] to that tuple of types
    return type(types)
"""
R = make_ordered_score_tuple([Rubric(desc="ropeproject", ge=0, le=10),
                             Rubric(desc="ropeproject", ge=20, le=30)])
"""

class Criteria(pydantic.BaseModel):
    rubrics: List[Rubric] = InputField(description=f"The rubrics used for grading the content")
    max_total_score: float = pydantic.Field(default=0.0, le=1.0, description="The sum of the rubric scores, but that must not exceed the maximum score")


class SemanticSignature[V](dspy.Signature):
    criteria: Criteria = InputField(description=f"The criteria for grading")
    input: V = InputField(description="The input to be graded")
    score: int = OutputField(description="The score of how the input meets the criteria")

T = TypeVar('T')
class GradingInput[T](TypedDict):
    criteria: Criteria
    input: T

C = TypeVar('C')

class GradingResult(pydantic.BaseModel):
    score:int

class GraderGenerationModule[C](dspy.Module):
    def __init__(self):
        self.grader = dspy.ChainOfThought(SemanticSignature[C])

    def forward(self, input: GradingInput[T]) -> GradingResult:
        return self.grader(**input)

class InterviewGenerationModule(dspy.Module):
    def __init__(self):
        self.analysis_plan = dspy.ChainOfThought(AnalysisPlanning)

    def forward(self, scenario: ScenarioArgs)->AnalysisPlanningResult:
        return self.analysis_plan(**scenario)

class SeeValStats(pydantic.BaseModel):
    histogram: List[List[int]] = pydantic.Field(default_factory=list, description="The histogram of the data")
    average_score: float = pydantic.Field(default=0.0, description="The average score of the data")
    max_score: int = pydantic.Field(default=0, description="The maximum score of the data")
    min_score: int = pydantic.Field(default=0, description="The minimum score of the data")
    total_score: int = pydantic.Field(default=0, description="The total score of the data")
    total_count: int = pydantic.Field(default=0, description="The total count of the data")

def make_export_json_type(DataModel: Type[pydantic.BaseModel])->Type[pydantic.BaseModel]:
    class ExportJSON(DataModel):
        score: int = pydantic.Field(default=0, description="The score of the data")

    class ExportJSONL(pydantic.BaseModel):
        seeval_stats: SeeValStats = pydantic.Field(description="The statistics of the data")
        data: List[ExportJSON] = pydantic.Field(default_factory=list, description="The data")
    return ExportJSONL

def get_single_page_instructions(DataModel: Type[pydantic.BaseModel], scale: bool, score=None, addendum=None):
    initial = f"""
    Create a single page dependency free html page, that is used to evaluate the success of a json object that is
    specified by the user. Here is the schema for the data being modeled
    The DataModel Schema: {DataModel.model_json_schema()}.

    Break this up logically that makes it easy to understand and navigate. The single page html file should allow you to import a .jsonl file that adheres to the schema
    and should provide a user interface for the user to interact with the data, by scoring it. We want to export the results to a .json file of the users choosing that results in a prefix that the user sets in an input box.
    we should also save the results of the evalution to storage so we can resume if we were in the middle of the evaluation. We should also be able to reset the evaluation.

    IMPORTING:
        The imported file should be a .jsonl file that contains the data to be evaluated. The file should be named according to the prefix set by the user in the input box. And the content should STRICTLY"
        follow {make_export_json_type(DataModel).model_json_schema()}

    EXPORTING:
        The exported file should be a .jsonl file that contains the results of the evaluation. The file should be named according to the prefix set by the user in the input box. And the content should STRICTLY"
        follow {make_export_json_type(DataModel).model_json_schema()}
    """
    new_score = f"""
    SCORING:
    Add a simple and easy to use rating mechanism that allows the user to accept or reject the data. This mechanism should be easy to use and should allow the user to quickly and easily rate the data.
    Keep track of the statistics of the ratings and provide a summary of the ratings at the end of the evaluation.
    """

    if scale:
        new_score = f"""
        SCORING:

        Add a simple and easy to use rating mechanism that allows the user to rate the data on a scale of 1 to 10.
        Keep track of the statistics of the ratings and provide a summary of the ratings at the end of the evaluation.
        """
    if(score is None):
        score = new_score

    return initial + score
