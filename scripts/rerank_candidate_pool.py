"""Rerank creative RAG candidate pool from build_reranker_pool_hf.py output.

Usage:
    python scripts/rerank_candidate_pool.py <input_jsonl>
    python scripts/rerank_candidate_pool.py  # uses newest *_reranker_pool_hf_*.jsonl
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def association_score(row: dict[str, Any]) -> float:
    au = row.get("association_use")
    if not isinstance(au, dict):
        return 0.0
    return safe_float(au.get("score_1_to_5"), 0.0)


WEIGHTS = {
    "min_score":       0.30,  # 1–10 scale
    "mean_score":      0.25,  # 1–10 scale
    "branch_fidelity": 0.20,  # 1–5 scale
    "association":     0.15,  # 1–5 scale
    "fixable_issue":   0.10,  # penalty, lower is better
}


def composite_score(row: dict[str, Any]) -> float:
    dm = row.get("derived_metrics") or {}
    return (
          WEIGHTS["min_score"]       * (safe_float(dm.get("min_score"))       / 10)
        + WEIGHTS["mean_score"]      * (safe_float(dm.get("mean_score"))      / 10)
        + WEIGHTS["branch_fidelity"] * (safe_float(dm.get("branch_fidelity")) / 5)
        + WEIGHTS["association"]     * (association_score(row)                / 5)
        - WEIGHTS["fixable_issue"]   * (int(dm.get("fixable_issue_count", 0)) / 10)
    )


def ranking_key(row: dict[str, Any]) -> float:
    return -composite_score(row)


def build_reason(row: dict[str, Any]) -> str:
    dm = row.get("derived_metrics") or {}
    parts = [
        f"violations={dm.get('constraint_violation_count', 0)}",
        f"min={safe_float(dm.get('min_score')):.1f}",
        f"mean={safe_float(dm.get('mean_score')):.1f}",
        f"fidelity={safe_float(dm.get('branch_fidelity')):.1f}",
        f"fixable={dm.get('fixable_issue_count', 0)}",
        f"assoc={association_score(row):.1f}",
        f"composite={composite_score(row):.3f}",
    ]
    return ", ".join(parts)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping line {line_no}: {exc}", file=sys.stderr)
    return records


def rerank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = rows[0].get("index") if rows else None

    drafted = [r for r in rows if r.get("status") == "drafted"]
    if not drafted:
        raise ValueError("No drafted rows found")

    # Keep only candidates with the minimum constraint_violation_count
    min_violations = min(
        int((r.get("derived_metrics") or {}).get("constraint_violation_count", 0))
        for r in drafted
    )
    candidates = [
        r for r in drafted
        if int((r.get("derived_metrics") or {}).get("constraint_violation_count", 0)) == min_violations
    ]

    ranked = sorted(candidates, key=ranking_key)

    selected = ranked[0]
    ranking_list = [
        {
            "rank": i + 1,
            "pool_variant": r.get("pool_variant"),
            "branch_id": r.get("branch_id"),
            "score": round(composite_score(r), 4),
            "reason": build_reason(r),
        }
        for i, r in enumerate(ranked)
    ]

    return {
        "index": index,
        "selected": {
            "pool_variant": selected.get("pool_variant"),
            "branch_id": selected.get("branch_id"),
            "candidate": selected.get("candidate", {}),
            "judgment": selected.get("judgment", {}),
            "derived_metrics": selected.get("derived_metrics", {}),
            "association_use": selected.get("association_use", {}),
        },
        "ranking": ranking_list,
    }


def output_path_for(input_path: Path) -> Path:
    stem = input_path.stem  # e.g. 180_reranker_pool_hf_20260531_175334
    parts = stem.split("_reranker_pool_hf_")
    index_part = parts[0] if parts else stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"{index_part}_reranker_result_{timestamp}.jsonl"


def latest_pool_path() -> Path:
    paths = sorted(
        RUNS_DIR.glob("*_reranker_pool_hf_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No *_reranker_pool_hf_*.jsonl found in {RUNS_DIR}")
    return paths[0]


def group_by_index(rows: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("index")
        groups.setdefault(key, []).append(row)
    return groups


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_pool_path()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    rows = iter_jsonl(input_path)
    if not rows:
        raise ValueError(f"No valid records in {input_path}")

    groups = group_by_index(rows)
    is_batch = len(groups) > 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_batch:
        out_path = RUNS_DIR / f"all_reranker_result_{timestamp}.jsonl"
    else:
        out_path = output_path_for(input_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    for index, group_rows in sorted(groups.items()):
        try:
            results.append(rerank(group_rows))
        except Exception as exc:
            errors.append((index, str(exc)))
            print(f"[WARN] index={index} skipped: {exc}", file=sys.stderr)

    with out_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Input : {input_path}")
    print(f"Output: {out_path}")
    print(f"Processed: {len(results)} indices" + (f", {len(errors)} errors" if errors else ""))
    if not is_batch:
        sel = results[0]["selected"]
        print(f"Selected: pool={sel['pool_variant']}, branch={sel['branch_id']}")
        print("Ranking:")
        for entry in results[0]["ranking"]:
            print(f"  #{entry['rank']} pool={entry['pool_variant']} branch={entry['branch_id']} | {entry['reason']}")


if __name__ == "__main__":
    main()
