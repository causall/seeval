import argparse
import json
from datetime import datetime
from typing import List

import dspy

from experiment.data_types import ExperimentConfig, ExperimentResult
from experiment.sentiment import data as exp_data
from experiment.sentiment import data_types as exp_types
from experiment.sentiment.core import (
    build_experiment_instances,
    build_split_datasets,
    build_split_examples,
    get_evaluation_criteria,
    run_gepa_experiment,
    setup_experiment_lm,
)
from experiment.sentiment.storage import (
    complete_experiment_run,
    create_experiment_run,
    save_experiment_result,
    save_optimized_program,
)


def print_optimized_instructions(filepath: str):
    """Pretty print the optimized instructions from a saved program file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    instructions = (
        data.get("grader.predict", {}).get("signature", {}).get("instructions")
    )

    if instructions:
        print("\n" + "=" * 80)
        print("OPTIMIZED INSTRUCTIONS")
        print("=" * 80 + "\n")
        print(instructions)
        print("\n" + "=" * 80 + "\n")
    else:
        print(f"No instructions found in {filepath}")


def main():
    try:
        noise_params: List[float] = [0.0, 0.1, 0.2, 0.3]
        optimizations = ["light"]

        experiment_config = ExperimentConfig[exp_types.SentimentMetadata](
            model="openai/llama4-maverick",
            metadata=exp_types.SentimentMetadata(num_headlines=100),
        )

        lm = setup_experiment_lm(
            experiment_config.model,
            experiment_config.api_base,
            experiment_config.api_key,
        )
        dspy.configure(lm=lm)

        evaluation_dataset = exp_data.generate_evaluation_dataset(
            lm,
            experiment_config.metadata.num_headlines,
            experiment_config.seed,
        )

        personas = exp_data.generate_personas()
        persona_names = list(exp_types.Personas.model_fields.keys())

        run_id = create_experiment_run(
            experiment_config, noise_params, persona_names
        )
        print(f"Started experiment run: {run_id}")

        criteria = get_evaluation_criteria()

        instances = build_experiment_instances(
            noise_params, optimizations, seed=experiment_config.seed
        )

        for persona_name in persona_names:
            persona = getattr(personas, persona_name)
            for instance in instances:
                splits = build_split_datasets(
                    evaluation_dataset, persona, instance
                )
                examples = build_split_examples(splits, criteria, instance.seed)

                optimized_program, _teleprompter, baseline_score, optimized_score = (
                    run_gepa_experiment(lm, examples, criteria, instance.optimization)
                )
                print(
                    f"Persona: {persona_name}, noise={instance.noise}, "
                    f"baseline={baseline_score}, optimized={optimized_score}"
                )

                program_path = save_optimized_program(
                    optimized_program, run_id, persona_name, instance
                )

                result = ExperimentResult(
                    label=persona_name,
                    instance=instance,
                    baseline_score=baseline_score,
                    optimized_score=optimized_score,
                    optimized_program_path=program_path,
                    timestamp=datetime.now(),
                )
                save_experiment_result(result, run_id)

        complete_experiment_run(run_id)
        print(f"Completed experiment run: {run_id}")

    except Exception as e:
        print(f"Error: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment experiment runner")
    parser.add_argument(
        "--print-instructions",
        "-p",
        type=str,
        metavar="FILE",
        help="Pretty print optimized instructions from a saved program file",
    )

    args = parser.parse_args()

    if args.print_instructions:
        print_optimized_instructions(args.print_instructions)
    else:
        main()
