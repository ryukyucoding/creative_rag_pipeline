"""Build a reranker-ready candidate pool with multiple RAG variants.

This script is the handoff entry point for a reranker teammate. It runs the
shared ToT setup once, generates candidates for multiple retrieval variants,
judges each candidate, computes deterministic metrics, and writes one JSONL row
per candidate. It intentionally stops before programmatic rerank and final
revision.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from creative_rag import build_association_packet, format_packet_for_prompt, load_kb
from rerank_tot_candidats import derive_judgment_metrics
from run_tot_hf import (
    CANDIDATE_RETRIES,
    DRAFT_FORMAT_INSTRUCTIONS,
    GENERATION_MODEL_ID,
    HFChatClient,
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_MODEL_ID,
    JUDGE_TEMPERATURE,
    MAX_NEW_TOKENS,
    MIN_CANDIDATE_CHARS,
    ProgressBar,
    RAG_TOP_K,
    TEMPERATURE,
    build_judge_user,
    build_retry_instruction,
    candidate_validation_errors,
    fill_template,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"

REQUEST_PATH: Optional[Path] = None
RAG_VARIANTS = [
    item.strip()
    for item in os.environ.get(
        "RAG_VARIANTS",
        "none,random_association,near_relevance,creative_association",
    ).split(",")
    if item.strip()
]


def latest_request_path() -> Path:
    request_paths = sorted(
        RUNS_DIR.glob("*_request_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not request_paths:
        raise FileNotFoundError(
            f"No *_request_*.jsonl files found in {RUNS_DIR}. Run build_tot_requests.py first."
        )
    return request_paths[0]


def output_path_for(records: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(records) == 1:
        index = records[0].get("index", "unknown")
        return RUNS_DIR / f"{index}_reranker_pool_hf_{timestamp}.jsonl"
    return RUNS_DIR / f"batch_reranker_pool_hf_{timestamp}.jsonl"


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def repair_candidate_from_text(text: str, branch_id: Any) -> Optional[dict[str, Any]]:
    """Best-effort repair for models that emit raw multiline JSON strings."""
    if not text.strip():
        return None

    branch_match = re.search(r'"branch_id"\s*:\s*"([^"]+)"', text)
    answer_match = re.search(
        r'"candidate_answer"\s*:\s*"(?P<answer>.*?)(?:"\s*,\s*"self_check_notes"|"self_check_notes"\s*:)',
        text,
        flags=re.DOTALL,
    )
    if not answer_match:
        return None

    answer = answer_match.group("answer")
    answer = answer.replace('\\"', '"').replace("\\n", "\n").strip()
    answer = re.sub(r'"\s*,\s*$', "", answer).strip()
    if len(answer) < MIN_CANDIDATE_CHARS:
        return None

    return {
        "branch_id": branch_match.group(1) if branch_match else str(branch_id),
        "candidate_answer": answer,
        "self_check_notes": [
            "Candidate was repaired from malformed JSON emitted by the model."
        ],
    }


def draft_candidate(
    record: dict[str, Any],
    stages: dict[str, Any],
    client: HFChatClient,
    constraint_map: dict[str, Any],
    branch: dict[str, Any],
    association_packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    branch_id = branch.get("id")
    draft_user = (
        fill_template(
            stages["draft_branch"]["user_template"],
            {
                "constraint_map_json": constraint_map,
                "branch_plan_json": branch,
            },
        )
        + (
            format_packet_for_prompt(association_packet)
            if association_packet.get("mode") != "none"
            else ""
        )
        + DRAFT_FORMAT_INSTRUCTIONS
    )

    candidate: Any = None
    errors: list[str] = []
    for attempt in range(CANDIDATE_RETRIES + 1):
        if attempt > 0:
            print(
                f"  - retry draft variant={association_packet.get('mode')} branch={branch_id}",
                flush=True,
            )
        try:
            candidate = client.chat(
                GENERATION_MODEL_ID,
                stages["draft_branch"]["system"],
                draft_user if attempt == 0 else draft_user + build_retry_instruction(errors),
                TEMPERATURE if attempt == 0 else min(TEMPERATURE, 0.3),
                MAX_NEW_TOKENS,
            )
        except ValueError as exc:
            repaired = repair_candidate_from_text(client.last_text, branch_id)
            if repaired is not None:
                candidate = repaired
                errors = []
                break
            candidate = None
            errors = [f"draft did not return valid JSON: {exc}"]
            continue

        errors = candidate_validation_errors(candidate, MIN_CANDIDATE_CHARS)
        if not errors:
            break

    retry_note = {
        "attempts": attempt + 1,
        "validation_errors": errors,
    }
    if errors:
        raise ValueError(
            "Draft failed for "
            f"index={record.get('index')} variant={association_packet.get('mode')} "
            f"branch={branch_id}: {errors}"
        )
    return candidate, retry_note


def judge_candidate(
    stages: dict[str, Any],
    client: HFChatClient,
    candidate: dict[str, Any],
    association_packet: dict[str, Any],
) -> dict[str, Any]:
    judge_user = build_judge_user(
        stages["judge_branch"]["user_template"],
        candidate,
        association_packet,
    )
    return client.chat(
        JUDGE_MODEL_ID,
        stages["judge_branch"]["system"],
        judge_user,
        JUDGE_TEMPERATURE,
        JUDGE_MAX_NEW_TOKENS,
    )


def base_pool_row(
    record: dict[str, Any],
    variant: str,
    branch_id: Any,
    branch: dict[str, Any],
    association_packet: dict[str, Any],
    constraint_map: dict[str, Any],
    selected_branches: list[Any],
) -> dict[str, Any]:
    return {
        "index": record.get("index"),
        "domain1": record.get("domain1"),
        "domain2": record.get("domain2"),
        "lang": record.get("lang"),
        "query": record.get("query"),
        "generation_model": GENERATION_MODEL_ID,
        "judge_model": JUDGE_MODEL_ID,
        "pool_variant": variant,
        "branch_id": branch_id,
        "branch_plan": branch,
        "association_packet": association_packet,
        "constraint_map": constraint_map,
        "branch_expansion_summary": {
            "branch_count": len(selected_branches),
            "all_branch_ids": [
                item.get("id", i + 1) if isinstance(item, dict) else i + 1
                for i, item in enumerate(selected_branches)
            ],
        },
    }


def build_pool_for_record(
    record: dict[str, Any],
    client: HFChatClient,
    kb: list[dict[str, Any]],
    output_path: Path,
) -> int:
    if record.get("mode") != "staged":
        raise ValueError("build_reranker_pool_hf.py only supports staged request specs.")

    stages = record["stages"]
    branch_count = int(record.get("branch_count", 0) or 0)
    estimated_total = 2 + (len(RAG_VARIANTS) * max(1, branch_count) * 3)
    progress = ProgressBar(estimated_total, label=f"pool index={record.get('index')}")

    print(f"  - map constraints for index={record.get('index')}", flush=True)
    constraint_map = client.chat(
        GENERATION_MODEL_ID,
        stages["constraint_map"]["system"],
        stages["constraint_map"]["user"],
        TEMPERATURE,
        MAX_NEW_TOKENS,
    )
    progress.update("constraint map complete")

    print(f"  - expand branches for index={record.get('index')}", flush=True)
    expand_user = fill_template(
        stages["expand_branches"]["user_template"],
        {"constraint_map_json": constraint_map},
    )
    branch_expansion = client.chat(
        GENERATION_MODEL_ID,
        stages["expand_branches"]["system"],
        expand_user,
        TEMPERATURE,
        MAX_NEW_TOKENS,
    )
    branches = branch_expansion.get("branches", [])
    if not isinstance(branches, list) or not branches:
        raise ValueError(f"No branches returned for index={record.get('index')}")

    selected_branches = branches[: int(record.get("branch_count", len(branches)))]
    progress.update(f"expanded {len(branches)} branch plan(s)")

    query = str(record.get("query", ""))
    draft_rows: list[dict[str, Any]] = []
    written = 0

    for variant in RAG_VARIANTS:
        for branch_no, branch in enumerate(selected_branches, 1):
            branch_id = branch.get("id", branch_no) if isinstance(branch, dict) else branch_no
            print(f"  - retrieve variant={variant} branch={branch_id}", flush=True)
            association_packet = build_association_packet(
                query,
                constraint_map,
                branch,
                kb,
                RAG_TOP_K,
                variant,
            )
            progress.update(f"variant={variant} branch={branch_id} retrieval complete")

            print(f"  - draft variant={variant} branch={branch_id}", flush=True)
            row = base_pool_row(
                record,
                variant,
                branch_id,
                branch,
                association_packet,
                constraint_map,
                selected_branches,
            )
            try:
                candidate, retry_note = draft_candidate(
                    record,
                    stages,
                    client,
                    constraint_map,
                    branch,
                    association_packet,
                )
            except ValueError as exc:
                row["status"] = "draft_failed"
                row["candidate"] = None
                row["candidate_retry_note"] = {
                    "attempts": CANDIDATE_RETRIES + 1,
                    "validation_errors": [str(exc)],
                }
                row["error"] = str(exc)
                row["raw_model_text_preview"] = client.last_text[:1000]
                append_jsonl(output_path, row)
                written += 1
                progress.update(f"variant={variant} branch={branch_id} draft failed")
                progress.update(f"variant={variant} branch={branch_id} judgment skipped")
                continue

            row["status"] = "drafted"
            row["branch_id"] = candidate.get("branch_id", branch_id)
            row["candidate"] = candidate
            row["candidate_retry_note"] = retry_note
            draft_rows.append(row)
            progress.update(f"variant={variant} branch={branch_id} draft complete")

    for row in draft_rows:
        print(
            f"  - judge variant={row['pool_variant']} branch={row['branch_id']}",
            flush=True,
        )
        judgment = judge_candidate(
            stages,
            client,
            row["candidate"],
            row["association_packet"],
        )
        row["judgment"] = judgment
        row["derived_metrics"] = derive_judgment_metrics(judgment)
        row["association_use"] = judgment.get("association_use", {})
        append_jsonl(output_path, row)
        written += 1
        progress.update(
            f"variant={row['pool_variant']} branch={row['branch_id']} judgment complete"
        )

    return written


def main() -> None:
    input_path = REQUEST_PATH or latest_request_path()
    records = load_jsonl(input_path)
    output_path = output_path_for(records)
    kb = load_kb()
    client = HFChatClient()

    total_written = 0
    print(
        "Building reranker candidate pool with variants: "
        + ", ".join(RAG_VARIANTS),
        flush=True,
    )
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] index={record.get('index')}", flush=True)
        total_written += build_pool_for_record(record, client, kb, output_path)

    print(f"Read staged ToT request(s) from: {input_path}")
    print(f"Wrote {total_written} reranker candidate row(s) to: {output_path}")


if __name__ == "__main__":
    main()
