"""Storage utilities for sentiment experiment runs.

Saves optimized programs, results, and manifests to a `runs/<run_id>/` folder
structure. Now uses the generic `experiment.data_types` types so the on-disk
schema is shared with any future experiments built on the same shape (label =
persona_name for sentiment).
"""

from datetime import datetime
from pathlib import Path
from typing import List

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
    return RUNS_DIR / run_id


def _get_program_filename(label: str, instance: ExperimentInstance) -> str:
    return f"{label}_{instance.noise}_{instance.optimization}.json"


def _generate_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def create_experiment_run(
    config: ExperimentConfig,
    noise_params: List[float],
    labels: List[str],
) -> str:
    """Create a new experiment run directory and manifest.

    Args:
        config: The experiment configuration
        noise_params: List of noise parameters used
        labels: List of run labels (persona names for sentiment)

    Returns:
        The run_id for this experiment run
    """
    run_id = _generate_run_id()
    run_dir = _get_run_dir(run_id)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "optimized").mkdir(exist_ok=True)

    manifest = ExperimentRunManifest(
        run_id=run_id,
        config=config,
        noise_params=noise_params,
        created_at=datetime.now(),
        labels=labels,
    )

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    results = ExperimentRunResults(run_id=run_id, results=[])
    results_path = run_dir / "results.json"
    results_path.write_text(results.model_dump_json(indent=2))

    return run_id


def save_optimized_program(
    program: dspy.Module,
    run_id: str,
    label: str,
    instance: ExperimentInstance,
) -> str:
    """Save an optimized dspy program to the run's optimized folder.

    Returns:
        Relative path (from the run directory) to the saved program file.
    """
    run_dir = _get_run_dir(run_id)
    filename = _get_program_filename(label, instance)
    program_path = run_dir / "optimized" / filename

    program.save(str(program_path))

    return f"optimized/{filename}"


def save_experiment_result(result: ExperimentResult, run_id: str) -> None:
    """Append an experiment result to the run's results file."""
    run_dir = _get_run_dir(run_id)
    results_path = run_dir / "results.json"

    existing = ExperimentRunResults.model_validate_json(results_path.read_text())
    existing.results.append(result)
    results_path.write_text(existing.model_dump_json(indent=2))


def complete_experiment_run(run_id: str) -> None:
    """Mark an experiment run as completed by setting completed_at timestamp."""
    run_dir = _get_run_dir(run_id)
    manifest_path = run_dir / "manifest.json"

    manifest = ExperimentRunManifest.model_validate_json(manifest_path.read_text())
    manifest.completed_at = datetime.now()
    manifest_path.write_text(manifest.model_dump_json(indent=2))


def load_experiment_run(
    run_id: str,
) -> tuple[ExperimentRunManifest, ExperimentRunResults]:
    """Load an experiment run's manifest and results."""
    run_dir = _get_run_dir(run_id)

    manifest = ExperimentRunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text()
    )
    results = ExperimentRunResults.model_validate_json(
        (run_dir / "results.json").read_text()
    )

    return manifest, results


def list_experiment_runs() -> List[str]:
    """List all experiment run IDs (newest first)."""
    if not RUNS_DIR.exists():
        return []
    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)
