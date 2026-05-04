"""Serialization helpers for producing/consuming per-cohort dspy.Example shards.

Shards are JSONL files, one record per line, containing only the per-example
payload (`input` + `scores`). The shared `criteria` is stored once in the
manifest to keep shard size small and to avoid repeated JSON on every line.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dspy
import pydantic

import seevals.data_types as types


class ExampleRecord(pydantic.BaseModel):
    """One example's serializable payload. Criteria lives in the manifest."""
    input: dict
    scores: List[types.ScoredRubric] = pydantic.Field(default_factory=list)


class ShardManifest(pydantic.BaseModel):
    """Manifest describing a produced shard directory."""
    schema_version: int = 1
    criteria: types.Criteria
    seed: int
    produced_indices: List[int] = pydantic.Field(default_factory=list)
    exp_valid_movie_count: int = 0
    total_examples: Optional[int] = None
    source_config: Optional[str] = None
    created_at: datetime = pydantic.Field(default_factory=datetime.utcnow)


def _example_to_record(ex: dspy.Example) -> ExampleRecord:
    data = ex.toDict() if hasattr(ex, "toDict") else dict(ex)

    input_data = data.get("input")
    if input_data is None:
        raise ValueError("Example is missing 'input'")

    raw_scores = data.get("scores") or []
    # eval_data_to_examples stores scores as [ [ScoredRubric, ...] ]
    flat: List[types.ScoredRubric] = []
    if raw_scores and isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
        for inner in raw_scores:
            for s in inner:
                flat.append(_coerce_scored_rubric(s))
    else:
        for s in raw_scores:
            flat.append(_coerce_scored_rubric(s))

    return ExampleRecord(input=input_data, scores=flat)


def _coerce_scored_rubric(s) -> types.ScoredRubric:
    if isinstance(s, types.ScoredRubric):
        return s
    if isinstance(s, dict):
        return types.ScoredRubric.model_validate(s)
    # pydantic-like with model_dump
    if hasattr(s, "model_dump"):
        return types.ScoredRubric.model_validate(s.model_dump())
    raise TypeError(f"Unsupported score entry type: {type(s)!r}")


def _record_to_example(rec: ExampleRecord, criteria: types.Criteria) -> dspy.Example:
    kwargs = {"criteria": criteria, "input": rec.input}
    if rec.scores:
        kwargs["scores"] = [list(rec.scores)]
    return dspy.Example(**kwargs).with_inputs("criteria", "input")


def write_examples_shard(path: Path, examples: List[dspy.Example]) -> int:
    """Write examples as JSONL. Returns number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            rec = _example_to_record(ex)
            f.write(rec.model_dump_json())
            f.write("\n")
            count += 1
    return count


def load_examples_shard(path: Path, criteria: types.Criteria) -> List[dspy.Example]:
    """Stream-load a shard JSONL into dspy.Examples attached to `criteria`."""
    examples: List[dspy.Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = ExampleRecord.model_validate_json(line)
            examples.append(_record_to_example(rec, criteria))
    return examples


def write_manifest(path: Path, manifest: ShardManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def load_manifest(path: Path) -> ShardManifest:
    return ShardManifest.model_validate_json(path.read_text())


def shard_path(out_dir: Path, split: str, sample_idx: int) -> Path:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train|val|test, got {split!r}")
    return out_dir / f"{split}_{sample_idx}.jsonl"


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"
