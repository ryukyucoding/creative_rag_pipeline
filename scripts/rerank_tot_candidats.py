"""Programmatically rerank Tree-of-Thought candidate judgments.

Edit INPUT_PATH below to rerank a specific JSONL file. Leave it as None to use
the newest ``*_output_*.jsonl`` file under data/runs/. The script accepts either
one judgment per line or bundle records with a ``judgments`` list.
"""


import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"

# Set this to a concrete Path to rerank a specific JSONL file. Leave it as None
# to rerank the newest run_tot_ollama.py output file.
INPUT_PATH: Optional[Path] = None


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


def derive_judgment_metrics(judgment: dict[str, Any]) -> dict[str, Any]:
    scores = []
    for item in judgment.get("scores_by_criterion", []):
        if not isinstance(item, dict):
            scores.append(0.0)
            continue
        scores.append(safe_float(item.get("score_1_to_10")))

    branch_fidelity = judgment.get("branch_fidelity", {})
    if isinstance(branch_fidelity, dict):
        branch_fidelity_score = safe_float(branch_fidelity.get("score_1_to_5"))
    else:
        branch_fidelity_score = safe_float(branch_fidelity)

    issues = [
        item.get("main_issue", "")
        for item in judgment.get("scores_by_criterion", [])
        if isinstance(item, dict)
        and item.get("main_issue")
        and item.get("main_issue", "").lower() not in {"none", "n/a"}
    ]

    if isinstance(branch_fidelity, dict):
        bf_issue = branch_fidelity.get("main_issue")
        if bf_issue and bf_issue.lower() not in {"none", "n/a"}:
            issues.append(bf_issue)

    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "branch_fidelity": branch_fidelity_score,
        "constraint_violation_count": count_items(
            judgment.get("constraint_violations", [])
        ),
        "fixable_issue_count": len(issues),
        "fixable_issues": issues,
    }


def ranking_key(item: dict[str, Any]) -> tuple[int, float, float, float, int, str]:
    metrics = derive_judgment_metrics(item)
    return (
        metrics["constraint_violation_count"],
        -metrics["min_score"],
        -metrics["mean_score"],
        -metrics["branch_fidelity"],
        metrics["fixable_issue_count"],
        branch_sort_value(item.get("branch_id")),
    )


def rerank(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = sorted(judgments, key=ranking_key)
    selected = ranking[0] if ranking else None
    return {
        "selected_branch_id": None if selected is None else selected.get("branch_id"),
        "ranking": [
            {
                "branch_id": item.get("branch_id"),
                **derive_judgment_metrics(item),
            }
            for item in ranking
        ],
        "reason": (
            "Sorted by constraint_violations asc, min_score desc, "
            "mean_score desc, branch_fidelity desc, fixable_issue_count asc, "
            "branch_id asc. Metrics are derived from judge JSON."
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


def latest_output_path() -> Path:
    output_paths = sorted(
        RUNS_DIR.glob("*_output_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not output_paths:
        raise FileNotFoundError(
            f"No *_output_*.jsonl files found in {RUNS_DIR}. Run run_tot_ollama.py first."
        )
    return output_paths[0]


def output_path_for(outputs: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(outputs) == 1:
        index = outputs[0].get("index", "unknown")
        return RUNS_DIR / f"{index}_rerank_{timestamp}.jsonl"
    return RUNS_DIR / f"batch_rerank_{timestamp}.jsonl"


def write_outputs(outputs: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for item in outputs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    input_path = INPUT_PATH or latest_output_path()
    records = iter_jsonl(input_path)
    outputs = build_outputs(records)
    output_path = output_path_for(outputs)
    write_outputs(outputs, output_path)

    print(f"Read judge output(s) from: {input_path}")
    print(f"Wrote {len(outputs)} rerank result(s) to: {output_path}")


if __name__ == "__main__":
    main()
