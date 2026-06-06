"""Ablation evaluation over reranker results.

Reads all_reranker_result_*.jsonl (or a specified file) and produces:
  1. Per-variant average mean_score / min_score across all queries (ablation table)
  2. Per-subdomain breakdown for the selected candidate
  3. Constraint violation rate per variant

No LLM calls required — all data is already in the ranking array.

Usage:
    python scripts/evaluate_ablation.py
    python scripts/evaluate_ablation.py data/runs/all_reranker_result_20260603_171146.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"

VARIANTS = ["none", "random_association", "near_relevance", "creative_association"]


def latest_result_path() -> Path:
    paths = sorted(
        RUNS_DIR.glob("all_reranker_result_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No all_reranker_result_*.jsonl found in {RUNS_DIR}")
    return paths[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def parse_reason(reason: str) -> dict[str, float]:
    """Extract numeric fields from a reason string like 'violations=0, min=7.0, mean=7.4, ...'"""
    result: dict[str, float] = {}
    for key in ("violations", "min", "mean", "fidelity", "fixable", "assoc", "composite"):
        match = re.search(rf"{key}=([0-9.]+)", reason)
        if match:
            result[key] = float(match.group(1))
    return result


def collect_variant_scores(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    """For each variant, collect per-branch scores across all queries.

    Returns {variant: [{"mean": ..., "min": ..., "violations": ..., "assoc": ...}, ...]}.
    Each query contributes up to 3 entries (one per branch) per variant.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        for entry in record.get("ranking", []):
            variant = entry.get("pool_variant", "")
            parsed = parse_reason(entry.get("reason", ""))
            if "mean" in parsed:
                scores[variant].append(parsed)
    return scores


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_ablation_table(variant_scores: dict[str, list[dict[str, float]]]) -> None:
    print("\n=== Ablation Table (average across all queries × all branches) ===\n")
    header = f"{'Variant':<26} {'N':>4}  {'mean_score':>10}  {'min_score':>9}  {'assoc':>5}  {'violations%':>11}"
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        entries = variant_scores.get(variant, [])
        if not entries:
            print(f"{variant:<26}  {'—':>4}")
            continue
        n = len(entries)
        mean_scores = [e["mean"] for e in entries if "mean" in e]
        min_scores = [e["min"] for e in entries if "min" in e]
        assoc_scores = [e["assoc"] for e in entries if "assoc" in e]
        violation_flags = [1 if e.get("violations", 0) > 0 else 0 for e in entries]

        print(
            f"{variant:<26} {n:>4}  "
            f"{avg(mean_scores):>10.3f}  "
            f"{avg(min_scores):>9.3f}  "
            f"{avg(assoc_scores):>5.2f}  "
            f"{100 * avg(violation_flags):>10.1f}%"
        )


def print_selected_summary(records: list[dict[str, Any]]) -> None:
    """Stats for the actually-selected (rank-1) candidates."""
    print("\n=== Selected Candidates Summary ===\n")

    mean_scores = []
    min_scores = []
    violation_count = 0
    variant_counts: dict[str, int] = defaultdict(int)
    subdomain_scores: dict[str, list[float]] = defaultdict(list)

    for record in records:
        sel = record.get("selected", {})
        dm = sel.get("derived_metrics", {})
        variant = sel.get("pool_variant", "unknown")
        variant_counts[variant] += 1

        mean = dm.get("mean_score")
        mn = dm.get("min_score")
        violations = dm.get("constraint_violation_count", 0)

        if mean is not None:
            mean_scores.append(float(mean))
        if mn is not None:
            min_scores.append(float(mn))
        if violations:
            violation_count += 1

        subdomain = record.get("selected", {}).get("candidate", {})
        # subdomain comes from the original query — not always in selected.
        # Fall back to checking the ranking entries for domain info.
        # (domain2 is stored on the pool row but not copied into selected)

    print(f"Total queries evaluated : {len(records)}")
    print(f"mean_score (corpus avg) : {avg(mean_scores):.3f} / 10")
    print(f"min_score  (corpus avg) : {avg(min_scores):.3f} / 10")
    print(f"Constraint violations   : {violation_count} / {len(records)} ({100*violation_count/max(len(records),1):.1f}%)")

    print("\nSelected variant distribution:")
    for variant in VARIANTS:
        count = variant_counts.get(variant, 0)
        bar = "#" * count
        print(f"  {variant:<26} {count:>3}  {bar}")


def print_per_query(records: list[dict[str, Any]]) -> None:
    print("\n=== Per-Query Selected Scores ===\n")
    print(f"{'index':>6}  {'variant':<26}  {'br':>2}  {'mean':>6}  {'min':>5}  {'viol':>4}")
    print("-" * 60)
    for record in sorted(records, key=lambda r: r.get("index", 0)):
        idx = record.get("index", "?")
        sel = record.get("selected", {})
        dm = sel.get("derived_metrics", {})
        variant = sel.get("pool_variant", "?")
        branch = sel.get("branch_id", "?")
        mean = dm.get("mean_score", "?")
        mn = dm.get("min_score", "?")
        viol = dm.get("constraint_violation_count", 0)
        print(f"{idx:>6}  {variant:<26}  {branch:>2}  {mean:>6}  {mn:>5}  {viol:>4}")


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_result_path()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    print(f"Input: {input_path}")
    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    print(f"Loaded {len(records)} query results")

    variant_scores = collect_variant_scores(records)
    print_ablation_table(variant_scores)
    print_selected_summary(records)
    print_per_query(records)


if __name__ == "__main__":
    main()
