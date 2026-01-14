"""Storage utilities for experiment runs.

Saves optimized programs, results, and manifests to a runs/ folder structure.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dspy

from experiment.data_types import (
    ExperimentConfig,
    ExperimentInstance,
    ExperimentResult,
    ExperimentRunManifest,
    ExperimentRunResults,
)

RUNS_DIR = Path("runs")


def _get_run_dir(run_id: str) -> Path:
    """Get the directory path for a run."""
    return RUNS_DIR / run_id


def _get_program_filename(persona_name: str, instance: ExperimentInstance) -> str:
    """Generate filename for optimized program."""
    return f"{persona_name}_{instance.noise}_{instance.optimization}.json"


def _generate_run_id() -> str:
    """Generate a unique run ID using timestamp."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def create_experiment_run(
    config: ExperimentConfig,
    noise_params: List[float],
    persona_names: List[str],
) -> str:
    """Create a new experiment run directory and manifest.

    Args:
        config: The experiment configuration
        noise_params: List of noise parameters used
        persona_names: List of persona names being tested

    Returns:
        The run_id for this experiment run
    """
    run_id = _generate_run_id()
    run_dir = _get_run_dir(run_id)

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "optimized").mkdir(exist_ok=True)

    # Create manifest
    manifest = ExperimentRunManifest(
        run_id=run_id,
        config=config,
        noise_params=noise_params,
        created_at=datetime.now(),
        persona_names=persona_names,
    )

    # Write manifest
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    # Initialize empty results file
    results = ExperimentRunResults(run_id=run_id, results=[])
    results_path = run_dir / "results.json"
    results_path.write_text(results.model_dump_json(indent=2))

    return run_id


def save_optimized_program(
    program: dspy.Module,
    run_id: str,
    persona_name: str,
    instance: ExperimentInstance,
) -> str:
    """Save an optimized dspy program to the run's optimized folder.

    Args:
        program: The optimized dspy module
        run_id: The experiment run ID
        persona_name: Name of the persona (e.g., "positive", "negative")
        instance: The experiment instance configuration

    Returns:
        Relative path to the saved program file
    """
    run_dir = _get_run_dir(run_id)
    filename = _get_program_filename(persona_name, instance)
    program_path = run_dir / "optimized" / filename

    # Save using dspy's native save (state only, JSON format)
    program.save(str(program_path), save_program=False)

    # Return relative path from run directory
    return f"optimized/{filename}"


def save_experiment_result(result: ExperimentResult, run_id: str) -> None:
    """Append an experiment result to the run's results file.

    Args:
        result: The experiment result to save
        run_id: The experiment run ID
    """
    run_dir = _get_run_dir(run_id)
    results_path = run_dir / "results.json"

    # Load existing results
    existing = ExperimentRunResults.model_validate_json(
        results_path.read_text())

    # Append new result
    existing.results.append(result)

    # Write back
    results_path.write_text(existing.model_dump_json(indent=2))


def complete_experiment_run(run_id: str) -> None:
    """Mark an experiment run as completed by setting completed_at timestamp.

    Args:
        run_id: The experiment run ID
    """
    run_dir = _get_run_dir(run_id)
    manifest_path = run_dir / "manifest.json"

    # Load manifest
    manifest = ExperimentRunManifest.model_validate_json(
        manifest_path.read_text())

    # Update completed_at
    manifest.completed_at = datetime.now()

    # Write back
    manifest_path.write_text(manifest.model_dump_json(indent=2))


def load_experiment_run(run_id: str) -> tuple[ExperimentRunManifest, ExperimentRunResults]:
    """Load an experiment run's manifest and results.

    Args:
        run_id: The experiment run ID

    Returns:
        Tuple of (manifest, results)
    """
    run_dir = _get_run_dir(run_id)

    manifest = ExperimentRunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text()
    )
    results = ExperimentRunResults.model_validate_json(
        (run_dir / "results.json").read_text()
    )

    return manifest, results


def list_experiment_runs() -> List[str]:
    """List all experiment run IDs.

    Returns:
        List of run IDs sorted by date (newest first)
    """
    if not RUNS_DIR.exists():
        return []

    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)
