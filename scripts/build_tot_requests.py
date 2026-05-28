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
Your job is to maximize the supplied WritingBench checklist while preserving
literary quality, originality, and the user's requested form."""


EXPECTED_SINGLE_CALL_SCHEMA: dict[str, Any] = {
    "constraint_map": {
        "hard_requirements": ["..."],
        "soft_preferences": ["..."],
        "likely_failure_modes": ["..."],
    },
    "branches": [
        {
            "id": "A",
            "creative_intent": "...",
            "form_strategy": "...",
            "imagery_or_materials": ["..."],
            "checklist_coverage": ["..."],
            "risk": "...",
            "score_estimate": 0,
        }
    ],
    "selection": {
        "chosen_branch_ids": ["..."],
        "rationale": "Brief rubric-grounded reason, not private reasoning.",
    },
    "final_answer": "...",
    "rubric_audit": [
        {
            "criterion": "...",
            "score_1_to_10": 0,
            "evidence": "...",
            "remaining_risk": "...",
        }
    ],
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
            "Instruct, compact checklist, staged mode, simple LLM judge, and "
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
        help="Include full rubric JSON instead of compact criterion summaries.",
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
        "checklist": checklist_payload(entry, include_full_checklist),
        "branch_count": branch_count,
    }
    return f"""Run a lightweight Tree-of-Thought creative-writing pass.

Process:
1. Build a concise constraint map from the query and checklist.
2. Propose exactly {branch_count} distinct creative branches. Vary form, stance,
   imagery, narrative angle, or rhetorical strategy when useful.
3. Score the branches against the checklist and identify the best branch or blend.
4. Write the final answer requested by the user.
5. Audit the final answer against every checklist criterion.

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
    base_payload = {
        "query_id": entry.get("index"),
        "domain": entry.get("domain1"),
        "subdomain": entry.get("domain2"),
        "language": entry.get("lang"),
        "query": entry.get("query"),
        "checklist": checklist_payload(entry, include_full_checklist),
    }
    base_json = json.dumps(base_payload, ensure_ascii=False, indent=2)

    return {
        "expand": {
            "system": TOT_SYSTEM_PROMPT,
            "user": f"""Create exactly {branch_count} distinct branch plans for this creative-writing task.
Each branch must be compact and actionable. Return JSON with:
constraint_map, branches[id, creative_intent, form_strategy, imagery_or_materials,
checklist_coverage, risk].

Input:
{base_json}
""",
        },
        "draft_branch": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Write one candidate answer for the selected branch.
Respect the original user query and all hard constraints. Use the branch plan as
guidance, not as text to copy. Return exactly one JSON object with only these
keys: branch_id, candidate_answer, self_check_notes. candidate_answer must be a
complete, directly usable answer as one plain string. Do not use outline headings
or act numbers as JSON keys; put all headings and bullets inside candidate_answer.

Original input:
{base_json}

Branch plan:
{{branch_plan_json}}
""",
        },
        "judge_branch": {
            "system": TOT_SYSTEM_PROMPT,
            "judge_mode": judge_mode,
            "user_template": f"""Use a simple rubric judge for one candidate answer.
Score each checklist criterion from 1 to 10. Keep the critique short and concrete.
Return JSON with:
branch_id, scores_by_criterion[name, score_1_to_10, evidence, main_issue],
mean_score, min_score, constraint_violations, strengths, fixable_issues.

Original input:
{base_json}

Candidate:
{{candidate_json}}
""",
        },
        "synthesize": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Synthesize the final creative-writing answer from the best candidate material.
Use the programmatic rerank result and judge feedback to fix checklist weaknesses.
Return JSON with final_answer and rubric_audit. The final answer must be directly
usable by the original user.

Original input:
{base_json}

Programmatic rerank result:
{{rerank_result_json}}

Candidate judgments:
{{judgments_json}}

Candidate answers:
{{candidates_json}}
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
