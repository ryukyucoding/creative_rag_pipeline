"""Build Tree-of-Thought prompt requests for creative-writing evaluation.

The script is intentionally provider-neutral: it emits JSONL request/spec
objects that can be consumed by an OpenAI-compatible runner, a vLLM batch
runner, or a custom orchestration script.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "test_set" / "test_set_lit_arts_en.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "runs" / "tot_requests_lit_arts_en.jsonl"
LLAMA31_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"


TOT_SYSTEM_PROMPT = """You are a creative-writing Tree-of-Thought controller.
Use visible, concise deliberation artifacts only: constraint maps, branch plans,
rubric scores, and revision notes. Do not reveal hidden chain-of-thought.
During generation stages, follow only the user's query and extracted constraints.
Use the WritingBench checklist only when explicitly asked to judge or audit."""


CONSTRAINT_MAP_SCHEMA: dict[str, Any] = {
    "task_type": "...",
    "must_include": ["..."],
    "must_avoid": ["..."],
    "form_or_structure": ["..."],
    "style_or_tone": ["..."],
    "open_questions_or_ambiguities": ["..."],
    "likely_failure_modes": ["..."],
}


BRANCH_EXPANSION_SCHEMA: dict[str, Any] = {
    "branches": [
        {
            "id": "A",
            "name": "...",
            "narrative_architecture": "...",
            "voice_or_perspective": "...",
            "development_path": "...",
            "distinctive_material": "...",
            "risk": "...",
        }
    ]
}


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "branch_id": "A",
    "scores_by_criterion": [
        {
            "name": "...",
            "score_1_to_10": 0,
            "evidence": "...",
            "main_issue": "...",
        }
    ],
    "mean_score": 0.0,
    "min_score": 0,
    "constraint_violations": ["..."],
    "strengths": ["..."],
    "fixable_issues": ["..."],
}


FINAL_REVISION_SCHEMA: dict[str, Any] = {
    "final_answer": "...",
    "revision_notes": ["..."],
    "rubric_audit": [
        {
            "criterion": "...",
            "score_1_to_10": 0,
            "evidence": "...",
            "remaining_risk": "...",
        }
    ],
}


EXPECTED_SINGLE_CALL_SCHEMA: dict[str, Any] = {
    "constraint_map": CONSTRAINT_MAP_SCHEMA,
    "branches": BRANCH_EXPANSION_SCHEMA["branches"],
    "candidates": [
        {
            "branch_id": "A",
            "candidate_answer": "...",
            "self_check_notes": ["..."],
        }
    ],
    "judgments": [JUDGE_OUTPUT_SCHEMA],
    "selection": {
        "selected_branch_id": "A",
        "rationale": "Brief rubric-grounded reason, not private reasoning.",
    },
    "final_revision": FINAL_REVISION_SCHEMA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ToT prompt requests from the Lit & Arts EN test set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to WritingBench-style JSONL test set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path. Use '-' to print to stdout.",
    )
    parser.add_argument(
        "--preset",
        choices=("llama31-8b-staged",),
        default=None,
        help=(
            "Apply a recommended preset. llama31-8b-staged uses Llama 3.1 8B "
            "Instruct, staged mode, full-checklist judge, and "
            "programmatic rerank metadata."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model identifier to include in each request spec.",
    )
    parser.add_argument(
        "--mode",
        choices=("single-call", "staged"),
        default="single-call",
        help=(
            "single-call emits one complete ToT prompt per query; staged emits "
            "an orchestration spec with separate expansion/draft/judge/synthesis prompts."
        ),
    )
    parser.add_argument(
        "--branch-count",
        type=int,
        default=4,
        help="Number of ToT branches to ask the model to explore. Use 3 or 4 for 8B models.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("simple",),
        default="simple",
        help="LLM judge style for staged mode.",
    )
    parser.add_argument(
        "--rerank",
        choices=("programmatic", "none"),
        default="programmatic",
        help="Whether to include programmatic rerank policy metadata.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Zero-based offset in the input file.",
    )
    parser.add_argument(
        "--include-checklist-json",
        action="store_true",
        help="Use full rubric JSON for single-call mode. Staged judging always uses the full checklist.",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "llama31-8b-staged":
        args.model = args.model or LLAMA31_8B_INSTRUCT
        args.mode = "staged"
        args.judge_mode = "simple"
        args.rerank = "programmatic"
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def compact_checklist(entry: dict[str, Any]) -> list[dict[str, str]]:
    checklist = entry.get("checklist", [])
    return [
        {
            "name": str(item.get("name", "")),
            "criteria_description": str(item.get("criteria_description", "")),
            "excellent_band": str(item.get("9-10", "")),
        }
        for item in checklist
    ]


def checklist_payload(entry: dict[str, Any], include_full: bool) -> Any:
    if include_full:
        return entry.get("checklist", [])
    return compact_checklist(entry)


def build_single_call_user_prompt(
    entry: dict[str, Any],
    branch_count: int,
    include_full_checklist: bool,
) -> str:
    payload = {
        "query_id": entry.get("index"),
        "domain": entry.get("domain1"),
        "subdomain": entry.get("domain2"),
        "language": entry.get("lang"),
        "query": entry.get("query"),
        "checklist": entry.get("checklist", []),
        "branch_count": branch_count,
    }
    return f"""Run a lightweight Tree-of-Thought creative-writing pass.

Process:
1. Build a concise constraint map from the query only. Do not use the checklist.
2. Propose exactly {branch_count} complete creative branches. Each branch must be
   a different narrative architecture, not a checklist dimension.
3. Draft one complete candidate answer per branch using only the query,
   constraint map, and that branch.
4. Use the full checklist only now: judge each candidate and identify constraint
   violations.
5. Select the best candidate by constraint violations, min score, then mean score.
6. Revise only the selected candidate using its own judge feedback. Do not blend
   material from other candidates.

Output valid JSON matching this shape:
{json.dumps(EXPECTED_SINGLE_CALL_SCHEMA, ensure_ascii=False, indent=2)}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def build_staged_spec(
    entry: dict[str, Any],
    branch_count: int,
    include_full_checklist: bool,
    judge_mode: str,
) -> dict[str, Any]:
    generation_payload = {
        "query_id": entry.get("index"),
        "domain": entry.get("domain1"),
        "subdomain": entry.get("domain2"),
        "language": entry.get("lang"),
        "query": entry.get("query"),
    }
    judge_payload = {
        **generation_payload,
        "checklist": entry.get("checklist", []),
    }
    generation_json = json.dumps(generation_payload, ensure_ascii=False, indent=2)
    judge_json = json.dumps(judge_payload, ensure_ascii=False, indent=2)

    return {
        "constraint_map": {
            "system": TOT_SYSTEM_PROMPT,
            "user": f"""Extract a concise constraint map from the user's creative-writing query only.
Do not use or infer from any WritingBench checklist. Separate explicit
requirements from avoidances, style, structure, ambiguities, and likely failure
modes. Return exactly one JSON object matching this example shape:
{json.dumps(CONSTRAINT_MAP_SCHEMA, ensure_ascii=False, indent=2)}

Input:
{generation_json}
""",
        },
        "expand_branches": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Create exactly {branch_count} complete creative branches.
Each branch must be a different narrative architecture, not a different
checklist/rubric dimension. Do not score branches. Do not mention checklist
coverage. Keep each branch compact but complete enough to guide drafting.

Return exactly one JSON object matching this example shape:
{json.dumps(BRANCH_EXPANSION_SCHEMA, ensure_ascii=False, indent=2)}

Original input:
{generation_json}

Constraint map:
{{constraint_map_json}}
""",
        },
        "draft_branch": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Write one candidate answer for the selected branch.
Respect the original user query and the constraint map. Use the branch plan as
the narrative architecture, not as text to copy. Do not use any WritingBench
checklist or rubric during drafting.

Return exactly one JSON object with only these keys:
branch_id, candidate_answer, self_check_notes.
candidate_answer must be a complete, directly usable answer as one plain string.
Do not use outline headings or act numbers as JSON keys; put all headings and
bullets inside candidate_answer.

Original input:
{generation_json}

Constraint map:
{{constraint_map_json}}

Branch plan:
{{branch_plan_json}}
""",
        },
        "judge_branch": {
            "system": TOT_SYSTEM_PROMPT,
            "judge_mode": judge_mode,
            "user_template": f"""Use a simple rubric judge for one candidate answer.
Use the full WritingBench checklist below to score each criterion from 1 to 10.
Also check constraint violations against the original query requirements. Keep
the critique short and concrete.

Return exactly one JSON object matching this example shape:
{json.dumps(JUDGE_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}

Original input:
{judge_json}

Candidate:
{{candidate_json}}
""",
        },
        "final_revision": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Revise only the selected best candidate into the final answer.
Use the selected candidate and its judge feedback to fix weaknesses. Do not use,
quote, blend, or borrow from any other candidate.

Return exactly one JSON object matching this example shape:
{json.dumps(FINAL_REVISION_SCHEMA, ensure_ascii=False, indent=2)}
The final_answer must be directly usable by the original user.

Original input:
{judge_json}

Constraint map:
{{constraint_map_json}}

Programmatic rerank result:
{{rerank_result_json}}

Selected branch plan:
{{selected_branch_json}}

Selected judge feedback:
{{selected_judgment_json}}

Selected candidate:
{{selected_candidate_json}}
""",
        },
    }


def build_request(
    entry: dict[str, Any],
    mode: str,
    branch_count: int,
    include_full_checklist: bool,
    model: Optional[str],
    judge_mode: str,
    rerank: str,
) -> dict[str, Any]:
    common = {
        "index": entry.get("index"),
        "domain1": entry.get("domain1"),
        "domain2": entry.get("domain2"),
        "lang": entry.get("lang"),
        "query": entry.get("query"),
        "branch_count": branch_count,
        "mode": mode,
        "checklist_mode": "full" if include_full_checklist else "compact",
        "judge_checklist_mode": "full",
        "model": model,
    }

    if mode == "single-call":
        common["messages"] = [
            {"role": "system", "content": TOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_single_call_user_prompt(
                    entry,
                    branch_count,
                    include_full_checklist,
                ),
            },
        ]
        common["expected_output_schema"] = EXPECTED_SINGLE_CALL_SCHEMA
        return common

    common["judge_mode"] = judge_mode
    if rerank == "programmatic":
        common["programmatic_rerank_policy"] = {
            "primary_sort": "constraint_violations ascending",
            "secondary_sort": "min_score descending",
            "tertiary_sort": "mean_score descending",
            "tie_breaker": "prefer the candidate with fewer fixable_issues; if still tied, prefer lower branch_id",
            "selected_output_shape": {
                "selected_branch_id": "...",
                "ranking": [
                    {
                        "branch_id": "...",
                        "constraint_violations": 0,
                        "min_score": 0,
                        "mean_score": 0.0,
                    }
                ],
                "reason": "Computed from judge JSON, not by the LLM.",
            },
        }
    common["stages"] = build_staged_spec(
        entry,
        branch_count,
        include_full_checklist,
        judge_mode,
    )
    return common


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    if str(output) == "-":
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args = apply_preset(args)
    if args.branch_count < 1:
        raise ValueError("--branch-count must be >= 1")

    entries = load_jsonl(args.input)
    selected = entries[args.start_at:]
    if args.max_items is not None:
        selected = selected[:args.max_items]

    requests = [
        build_request(
            entry=entry,
            mode=args.mode,
            branch_count=args.branch_count,
            include_full_checklist=args.include_checklist_json,
            model=args.model,
            judge_mode=args.judge_mode,
            rerank=args.rerank,
        )
        for entry in selected
    ]
    write_jsonl(requests, args.output)

    if str(args.output) != "-":
        print(f"Wrote {len(requests)} ToT {args.mode} request(s) to: {args.output}")


if __name__ == "__main__":
    main()
