from pathlib import Path
import argparse
import json
from typing import List, Tuple

import jsonpath_ng as jp

from seevals.data_types import (
    Criteria,
    DataCriteria,
    DatumCriteria,
    EvalData,
    EvalDatum,
    EvalItem,
    View,
)
from seevals.utils import write_eval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scored EvalData from a JSONL log and a DataCriteria file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--criteria",
        metavar="FILENAME",
        type=str,
        required=True,
        help="DataCriteria JSON file describing rubrics + json_paths",
    )
    parser.add_argument(
        "--dataset",
        metavar="FILENAME",
        type=str,
        required=True,
        help="JSONL log file (one JSON object per line)",
    )
    parser.add_argument(
        "--output",
        metavar="FILENAME",
        type=str,
        required=True,
        help="Output path for the scored EvalData (.jsonl)",
    )
    return parser.parse_args()


def load_data_criteria(filename: str) -> DataCriteria:
    with open(filename, "r") as f:
        return DataCriteria.model_validate(json.load(f))


def load_jsonl(filename: str) -> List[dict]:
    entries: List[dict] = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_score(log_entry: dict, dc: DatumCriteria) -> float:
    if dc.json_path:
        matches = jp.parse(dc.json_path).find(log_entry)
        if matches:
            return float(matches[0].value)
    return dc.score


def build_eval_data(
    log_entries: List[dict],
    data_criteria: DataCriteria,
) -> Tuple[Criteria, List[EvalData]]:
    rubrics = [dc.rubric for dc in data_criteria.rubrics]
    criteria = Criteria(rubrics=rubrics)

    dataset: List[EvalData] = []
    for i, entry in enumerate(log_entries):
        datums: List[EvalDatum] = []
        for dc in data_criteria.rubrics:
            score = extract_score(entry, dc)
            item = EvalItem(
                id=dc.json_path or "$",
                sample=None,
                view=View(views=["$"]),
                data=entry,
                score=score,
            )
            datums.append(
                EvalDatum(
                    group_id=str(i),
                    items=[item],
                    rubric=dc.rubric,
                )
            )
        dataset.append(EvalData(data=datums, raw_data=entry))

    return criteria, dataset


def main():
    args = parse_args()

    data_criteria = load_data_criteria(args.criteria)
    log_entries = load_jsonl(args.dataset)
    criteria, eval_dataset = build_eval_data(log_entries, data_criteria)

    output_path = Path(args.output)
    write_eval_dataset(str(output_path), eval_dataset)

    criteria_path = output_path.with_suffix(".criteria.json")
    with open(criteria_path, "w") as f:
        f.write(criteria.model_dump_json(indent=2))

    print(f"Wrote {len(eval_dataset)} eval entries to {output_path}")
    print(f"Wrote criteria to {criteria_path}")


if __name__ == "__main__":
    main()
