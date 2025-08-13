import pydantic
import json
from typing import TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import dspy
import utils
import agent_util
import agents
from execute import run_parallel
import data_types as types
from agents import ScenarioArgs
from pdb import Pdb

T = TypeVar('T')
R = TypeVar('R', bound=pydantic.BaseModel)

lm = dspy.LM(
    model="openai/bedrock-sonnet-37",
    #model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
   # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
    api_base="http://localhost:4000",
    api_key="noop",
)


scenarios = utils.load_from("./scenarios.jsonl", agents.ScenarioArgs)
results = run_parallel(agents.InterviewGenerationModule(),scenarios, lm, 20)
rubrics = [
    types.Rubric(ge=0.0,le=0.30,desc="Plan overview is consistent with the entities and relationships"),
    types.Rubric(ge=0.0,le=0.25,desc="Relationships make logical sense with the scenario"),
    types.Rubric(ge=0.0,le=0.25,desc="Entities described have logical types"),
    types.Rubric(ge=0.0,le=0.10,desc="No entities that are not in a relationship"),
    types.Rubric(ge=0.0,le=0.10,desc="No relationships that contain entities that are not described")
]
criteria =types.Criteria(rubrics=rubrics,max_total_score=1)




#input_data=list(map(lambda val: val.analysis_plan, get_values(results.data)))
# grading_inputs = agent_util.make_grading_inputs(criteria,input_data)

results = utils.load_from_result('./results.jsonl', agents.AnalysisPlanningResult)
input_data=list(map(lambda val: val.analysis_plan, results))

grading_inputs = agent_util.make_grading_inputs(criteria,input_data)
grades = run_parallel(agents.GraderGenerationModule(),grading_inputs, lm, 20)


# generate synthetic data
# validate synthetic data
# use synthetic data to train model
# validate trained model on expected results



with open("./results.jsonl", 'w') as f:
    for value in get_values(results.data):
        f.write(json.dumps({"item": value.analysis_plan.model_dump()}) + "\n")

import pdb
pdb.set_trace()
print(results)
