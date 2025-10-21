import pdb
import pydantic
import json
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import dspy
import seevals.utils as utils
import seevals.agent_util as agent_util
import seevals.agents as agents
from seevals.execute import run_parallel
import seevals.data_types as types
from seevals.agents import ScenarioArgs
from pdb import Pdb

T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)

lm = dspy.LM(
    model="openai/bedrock-sonnet-37",
    # model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
    api_base="http://localhost:4000",
    api_key="noop",
)


scenarios = utils.load_from("./scenarios.jsonl", agents.ScenarioArgs)
results = run_parallel(agents.InterviewGenerationModule(), scenarios, lm, 1)
utils.write_results_from_response('./results.jsonl', results)

rubrics = [
    types.Rubric(
        ge=0, le=30, desc="Plan overview is consistent with the entities and relationships"),
    types.Rubric(
        ge=0, le=25, desc="Relationships make logical sense with the scenario"),
    types.Rubric(ge=0, le=25, desc="Entities described have logical types"),
    types.Rubric(
        ge=0, le=10, desc="No entities that are not in a relationship"),
    types.Rubric(
        ge=0, le=10, desc="No relationships that contain entities that are not described")
]

rubrics = [
    types.Rubric(ge=0, le=2, desc="Plan overview is consistent with relationships",
                 scale="0 is not consistent, 1 is partially consistent, 2 is consistent"),
    types.Rubric(ge=0, le=3, desc="Entities described have logical types",
                 scale="0 is not logical, 1 is partially consistent, 2 is consistent, 3 is perfectly consistent"),
    types.Rubric(ge=0, le=3, desc="Relationships make logical sense",
                 scale="0 is not logical, 1 is partially logical, 2 is mostly logical, 3 is perfectly logical"),
]


eval = types.EvalDatasetBuilder.build(agents.AnalysisPlanningResult)

eval.add("$.analysis_overview", None,
         types.View(views=["$.analysis_indices.relationships_index", "$.analysis_indices.subsections_index"]), rubrics[0])

eval.add("$.analysis_indices.entities_index", types.Sample(
    num_samples=5), types.View(views=["$.analysis_overview"]), rubrics[1])

eval.add("$.analysis_indices.relationships_index", types.Sample(
    num_samples=5), types.View(views=["$.analysis_overview"]), rubrics[2])

evaluation_dataset = eval.apply(results.data, seed=47)

utils.write_eval_dataset('./evaluation_dataset.jsonl', evaluation_dataset)

pdb.set_trace()
"""
for each entry in the result, we extract the item,
then we reconstruct it using the config which looks like
an array of arrays now [[]] and each entry of that can then have a score then evaluation is just adding a score field to each
entry then. and simply iterating through a collection of these per item making it easy to color etc... in the ui

"""


"""
class EvalPlan:
    always_display: ["*",".",...]
    fields: [
        ""
    ]
    sample_fields: [
        {
            "path": "analysis_plan.entities",
            "num_samples": 10,
            "confidence": 0.95
        },
        {
            "path": "analysis_plan.relationships",
            "num_samples": 10,
            "confidence": 0.95
        },
    ]

"""

"""
sample_fields = [
    types.SampleCriteria(path="analysis_plan.entities",
                         model=AnalysisPlanningResult),
    types.SampleCriteria(path="analysis_plan.relationships",
                         model=AnalysisPlanningResult),
]
types.SampleCriteria(sample_fields=sample_fields, e=0.2, confidence=0.95)
"""


# def get_hoeffding_error_margin(confidence: float, num_samples: int) -> float:


# def make_sample_criteria(num_samples: int, confidence: float) -> types.SampleCriteria:

# SampleCriteria


criteria = types.Criteria(rubrics=rubrics, max_total_score=6)

# criteria =types.Criteria(rubrics=rubrics,max_total_score=100)


results = utils.load_from_result(
    './results.jsonl', agents.AnalysisPlanningResult)
input_data = list(map(lambda val: val, results))

grading_inputs = agent_util.make_grading_inputs(criteria, results)
grader = agents.make_semantic_grader(agents.AnalysisPlanningResult)
grades = run_parallel(grader, grading_inputs, lm, 20)

noise_factor = 0.1
contrastive_inputs = agents.from_grading_inputs(grading_inputs, noise_factor)

# you should be able to tag a sample on the field that you need to make a sample on for grading purposes
contraster = agents.make_contrastive_grader(agents.AnalysisPlanningResult)
contrastive_outputs = run_parallel(contraster, contrastive_inputs, lm, 20)

# sampler(contrastive_outputs, sample_criteria)


# generate synthetic data
# validate synthetic data
# use synthetic data to train model
# validate trained model on expected results


"""
with open("./results.jsonl", 'w') as f:
    for value in get_values(results.data):
        f.write(json.dumps({"item": value.analysis_plan.model_dump()}) + "\n")
"""
utils.write_results_from_response(
    './contrastive_outputs.jsonl', contrastive_outputs, criteria)
