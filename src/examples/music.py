from dspy.utils.callback import BaseCallback
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

# setup scaffolding


class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())


# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])

lm = dspy.LM(
    model="openai/bedrock-sonnet-37",
    # model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    # lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0')
    api_base="http://localhost:4000",
    api_key="noop",
)


# An llm that has some idea of my preferences for music evaluation

"""
The goal here is to be able to train a model that will generate a list of listenable music that you will rate. We also want to be able to go in the opposite
direction and evaluate a list of music and give a score for the listenability of the list.

"""


class Music(pydantic.BaseModel):
    title: str = pydantic.Field(description="The title of the song")
    artist: str = pydantic.Field(description="The artist of the song")
    year: int = pydantic.Field(description="The year of the song")
    genre: str = pydantic.Field(description="The genre of the song")


class MusicGenerationArgs(pydantic.BaseModel):
    num_tracks: int = pydantic.Field(
        description="The number of tracks to generate")


class Listenability(Music):
    listenability: int = pydantic.Field(
        description="The listenability of the song")


class MusicOutput(pydantic.RootModel):
    root: List[Music]


class MusicTrackGenerator(dspy.Signature):
    """
    You are a music generator, you generate a list of music for me to rate the listenability of create a mix of generes and years and it should be real music that exists.
    """
    num_tracks: int = dspy.InputField(
        description="The number of tracks to generate")
    tracks: MusicOutput = dspy.OutputField(
        description="The list of music")


class MusicGenerationModule(dspy.Module):
    def __init__(self):
        self.generator = dspy.ChainOfThought(
            MusicTrackGenerator)

    def forward(self, args: MusicGenerationArgs) -> dspy.Prediction:
        return self.generator(**args.model_dump())

    def get_value(self, prediction: dspy.Prediction) -> MusicOutput:
        return prediction.tracks


ListenabilityRubric = types.Rubric(id=1, ge=0, le=3, desc="Listenability of the track",
                                   scale="0 is not listenable, 1 is partially listenable, 2 is listenable, 3 is very listenable")


music_tracks = run_parallel(MusicGenerationModule(), [
                            MusicGenerationArgs(num_tracks=30)], lm, 1)

# we're just making our synthetic output match the format for the evaluation dataset
music_track_output = list(map(lambda x: Music(
    title=x['title'], artist=x['artist'], year=x['year'], genre=x['genre']), music_tracks.data[0].model_dump()))

eval = types.EvalDatasetBuilder.build(Music)

eval.add("$", None,
         types.View(views=["$"]), ListenabilityRubric)

evaluation_dataset = eval.apply(music_track_output, seed=45)

utils.write_eval_dataset(
    './music_evaluation_dataset.jsonl', evaluation_dataset)

grading_module = agents.make_semantic_grader(Music)
grading_inputs = agent_util.make_grading_inputs_from_eval_dataset(
    types.Criteria(rubrics=[ListenabilityRubric], scores=[]), evaluation_dataset)



pdb.set_trace()
grades = run_parallel(grading_module, grading_inputs, lm, 1)
