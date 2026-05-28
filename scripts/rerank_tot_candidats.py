"""Programmatically rerank Tree-of-Thought candidate judgments.

Expected inputs are JSONL records from the staged judge step. The script accepts
either one judgment per line or bundle records with a ``judgments`` list.
"""


import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank ToT candidates using simple deterministic rules."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file containing judge outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Omit or use '-' to print to stdout.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        return 0 if not value.strip() else 1
    if isinstance(value, (int, float)):
        return int(value)
    return 1


def branch_sort_value(branch_id: Any) -> str:
    return str(branch_id or "")


def normalize_judgment(record: dict[str, Any]) -> dict[str, Any]:
    scores = record.get("scores_by_criterion") or []
    derived_scores = [
        safe_float(item.get("score_1_to_10"))
        for item in scores
        if isinstance(item, dict)
    ]
    min_score = record.get("min_score")
    mean_score = record.get("mean_score")

    if min_score is None and derived_scores:
        min_score = min(derived_scores)
    if mean_score is None and derived_scores:
        mean_score = sum(derived_scores) / len(derived_scores)

    return {
        "branch_id": record.get("branch_id"),
        "constraint_violations": count_items(record.get("constraint_violations")),
        "min_score": safe_float(min_score),
        "mean_score": safe_float(mean_score),
        "fixable_issue_count": count_items(record.get("fixable_issues")),
        "raw_judgment": record,
    }


def ranking_key(item: dict[str, Any]) -> tuple[float, float, float, int, str]:
    return (
        safe_float(item.get("constraint_violations")),
        -safe_float(item.get("min_score")),
        -safe_float(item.get("mean_score")),
        int(item.get("fixable_issue_count", 0)),
        branch_sort_value(item.get("branch_id")),
    )


def rerank(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = sorted((normalize_judgment(j) for j in judgments), key=ranking_key)
    selected = ranking[0] if ranking else None
    return {
        "selected_branch_id": None if selected is None else selected.get("branch_id"),
        "ranking": [
            {
                "branch_id": item.get("branch_id"),
                "constraint_violations": item.get("constraint_violations"),
                "min_score": item.get("min_score"),
                "mean_score": item.get("mean_score"),
                "fixable_issue_count": item.get("fixable_issue_count"),
            }
            for item in ranking
        ],
        "reason": (
            "Sorted by constraint_violations asc, min_score desc, "
            "mean_score desc, fixable_issue_count asc, branch_id asc."
        ),
    }


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def build_outputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bundled: list[dict[str, Any]] = []
    loose: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if isinstance(record.get("judgments"), list):
            bundled.append(
                {
                    "index": record.get("index") or record.get("query_id"),
                    "rerank_result": rerank(record["judgments"]),
                }
            )
            continue

        key = str(record.get("index") or record.get("query_id") or "all")
        loose[key].append(record)

    for key, judgments in loose.items():
        bundled.append(
            {
                "index": None if key == "all" else key,
                "rerank_result": rerank(judgments),
            }
        )

    return bundled


def write_outputs(outputs: list[dict[str, Any]], output: Optional[Path]) -> None:
    if output is None or str(output) == "-":
        for item in outputs:
            print(json.dumps(item, ensure_ascii=False))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for item in outputs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = iter_jsonl(args.input)
    outputs = build_outputs(records)
    write_outputs(outputs, args.output)


if __name__ == "__main__":
    main()
