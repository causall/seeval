from datetime import datetime
from itertools import product
from attr import dataclass
from dspy.utils.callback import BaseCallback
import pdb
import pydantic
import json
import argparse
from typing import Literal, TypedDict, Type, Tuple, Dict, Iterable, ParamSpec, TypeVar, Generic, List, Callable, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import dspy
import seevals.utils as utils
import seevals.agent_util as agent_util
import seevals.agents as agents
from seevals.execute import run_parallel
import seevals.data_types as types
from seevals.agents import ScenarioArgs
import experiment.sentiment.utils as exp_utils
import experiment.sentiment.data_types as exp_types
import experiment.sentiment.data as exp_data
from experiment.sentiment.core import get_evaluation_criteria, setup_experiment_lm, run_experiment, build_experiment_instances
from experiment.sentiment.storage import (
    create_experiment_run,
    save_optimized_program,
    save_experiment_result,
    complete_experiment_run,
)


def print_optimized_instructions(filepath: str):
    """Pretty print the optimized instructions from a saved program file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract instructions from the nested structure
    instructions = data.get('grader.predict', {}).get(
        'signature', {}).get('instructions')

    if instructions:
        print("\n" + "="*80)
        print("OPTIMIZED INSTRUCTIONS")
        print("="*80 + "\n")
        print(instructions)
        print("\n" + "="*80 + "\n")
    else:
        print(f"No instructions found in {filepath}")


def generate_experiment_instances(evaluation_dataset, noise_params: List[float]):

    personas = exp_data.generate_personas()
    optimization = ["light"]

    pexp_instances = build_experiment_instances(evaluation_dataset, personas.positive,
                                                noise_params, optimization)
    nexp_instances = build_experiment_instances(evaluation_dataset, personas.negative,
                                                noise_params, optimization)
    nex_instances = build_experiment_instances(evaluation_dataset, personas.neutral,
                                               noise_params, optimization)
    ex_instances = build_experiment_instances(evaluation_dataset, personas.extreme,
                                              noise_params, optimization)
    eq_instances = build_experiment_instances(evaluation_dataset, personas.equal,
                                              noise_params, optimization)
    return {
        "positive": pexp_instances,
        "negative": nexp_instances,
        "neutral": nex_instances,
        "extreme": ex_instances,
        "equal": eq_instances
    }


def main():
    try:
        noise_params = [0.0, 0.1, 0.2, 0.3]
        experiment_config = exp_types.ExperimentConfig()
        experiment_config.model = "openai/gemma-3-27b"
        experiment_config.model = "openai/llama4-maverick"
        # setup the experiment lm
        lm = setup_experiment_lm(experiment_config.model,
                                 experiment_config.api_base, experiment_config.api_key)
        dspy.configure(lm=lm)

        # generate the evaluation dataset
        evaluation_dataset = exp_data.generate_evaluation_dataset(
            lm, experiment_config.num_headlines, experiment_config.seed)

        # generate test evaluation dataset
        test_evaluation_dataset = exp_data.generate_evaluation_dataset(
            lm, experiment_config.num_headlines, experiment_config.seed + 9)

        # generate the personas for the experiment

        # build the experiment instances for the positive persona

        experiment_instances = generate_experiment_instances(
            evaluation_dataset, noise_params)

        # Create experiment run and save manifest
        run_id = create_experiment_run(
            experiment_config,
            noise_params,
            list(experiment_instances.keys())
        )
        print(f"Started experiment run: {run_id}")

        for persona, instances in experiment_instances.items():
            for instance in instances:
                optimized_program, teleprompter, baseline_score, optimized_score = run_experiment(
                    lm, evaluation_dataset, test_evaluation_dataset, instance)
                print(
                    f"Persona: {persona}, Baseline score: {baseline_score}, Optimized score: {optimized_score}")

                # Save optimized program
                program_path = save_optimized_program(
                    optimized_program, run_id, persona, instance
                )

                # Save experiment result
                result = exp_types.ExperimentResult(
                    persona_name=persona,
                    instance=instance,
                    baseline_score=baseline_score,
                    optimized_score=optimized_score,
                    optimized_program_path=program_path,
                    timestamp=datetime.now()
                )
                save_experiment_result(result, run_id)

        # Mark run as completed
        complete_experiment_run(run_id)
        print(f"Completed experiment run: {run_id}")

    except Exception as e:
        print(f"Error: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment experiment runner')
    parser.add_argument('--print-instructions', '-p', type=str, metavar='FILE',
                        help='Pretty print optimized instructions from a saved program file')

    args = parser.parse_args()

    if args.print_instructions:
        print_optimized_instructions(args.print_instructions)
    else:
        main()
