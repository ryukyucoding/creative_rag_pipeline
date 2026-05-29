"""Build Tree-of-Thought prompt requests for creative-writing evaluation.

Edit the global configuration block below to choose the model, branch count,
and WritingBench test-set index. The script writes one staged request JSONL file
under data/runs/ using the selected index and current timestamp.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "test_set" / "test_set_lit_arts_en.jsonl"
RUNS_DIR = ROOT / "data" / "runs"

# Edit these values when you want to generate a different request file.
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BRANCH_COUNT = 3
TARGET_INDEX = 180

JUDGE_MODE = "simple"


TOT_SYSTEM_PROMPT = """You are a creative-writing Tree-of-Thought controller.
Use visible, concise deliberation artifacts only: constraint maps, branch plans,
rubric scores, and revision notes. Do not reveal hidden chain-of-thought."""


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
            "constraint_coverage": [
                {
                    "requirement": "...",
                    "plan": "...",
                }
            ],
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
    "branch_fidelity": {
        "score_1_to_5": 0,
        "evidence": "...",
        "main_issue": "...",
    },
    "constraint_violations": [],
    "top_fix": "...",
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


def build_staged_spec(entry: dict[str, Any]) -> dict[str, Any]:
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
Separate explicit requirements from avoidances, style, structure, ambiguities,
and likely failure modes. Return exactly one JSON object matching this example
shape:
{json.dumps(CONSTRAINT_MAP_SCHEMA, ensure_ascii=False, indent=2)}

Input:
{generation_json}
""",
        },
        "expand_branches": {
            "system": TOT_SYSTEM_PROMPT,
            "user_template": f"""Create exactly {BRANCH_COUNT} complete creative branches.
Each branch must be a complete solution path for the original task. Each branch
must cover every item in constraint_map.must_include, and constraint_coverage
must name each required item with a concrete plan for satisfying it. Do not split
required elements across separate branches: every branch must handle the whole
task. The branches must be genuinely different narrative architectures, not
minor variations or different evaluation dimensions. Do not score branches. Keep
each branch compact but complete enough to guide drafting.

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
Respect the original user query and the constraint map. Visibly execute the
selected branch's narrative_architecture. Do not fall back to a generic outline
or generic answer pattern. Use the branch's distinctive_material and follow its
development_path as the organizing logic for the candidate.

Return exactly one JSON object with only these keys:
branch_id, candidate_answer, self_check_notes.
candidate_answer must be a complete, directly usable answer as one plain string.
The JSON must be valid: escape line breaks inside candidate_answer as needed; do
not emit an unterminated string or raw multiline JSON string.
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
            "judge_mode": JUDGE_MODE,
            "user_template": f"""Use a simple rubric judge for one candidate answer.
Use the full WritingBench checklist below to score each criterion from 1 to 10.
Also score branch_fidelity from 1 to 5: how faithfully the candidate executes
its selected branch plan rather than falling back to a generic answer pattern.
Check constraint violations against the original query requirements. Keep the
critique short and concrete. Do not compute aggregate scores or list strengths;
the program derives aggregate metrics from this JSON.

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
Use the selected candidate, selected branch plan, selected judgment, and derived
metrics to fix weaknesses. Prioritize constraint_violations, top_fix, each
criterion's main_issue, and branch_fidelity.main_issue. Do not use, quote,
blend, or borrow from any other candidate.

The final_answer must contain a complete revised plot design, not just a title.
It must be at least as detailed as the selected_candidate.candidate_answer.
Do not shorten the selected candidate.
Do not output only the title.
Do not merely list suggestions.
Apply the selected judgment feedback directly inside final_answer.
revision_notes must describe edits actually made.

Return exactly one JSON object matching this example shape:
{json.dumps(FINAL_REVISION_SCHEMA, ensure_ascii=False, indent=2)}
The final_answer must be directly usable by the original user.

Original input:
{generation_json}

Constraint map:
{{constraint_map_json}}

Programmatic rerank result:
{{rerank_result_json}}

Final revision input:
{{final_revision_input_json}}
""",
        },
    }


def build_request(entry: dict[str, Any]) -> dict[str, Any]:
    common = {
        "index": entry.get("index"),
        "domain1": entry.get("domain1"),
        "domain2": entry.get("domain2"),
        "lang": entry.get("lang"),
        "query": entry.get("query"),
        "branch_count": BRANCH_COUNT,
        "mode": "staged",
        "judge_checklist_mode": "full",
        "model": MODEL,
    }

    common["judge_mode"] = JUDGE_MODE
    common["programmatic_rerank_policy"] = {
        "primary_sort": "constraint_violation_count ascending",
        "secondary_sort": "min_score descending",
        "tertiary_sort": "mean_score descending",
        "quaternary_sort": "branch_fidelity descending",
        "tie_breaker": "prefer fewer derived fixable issues; if still tied, prefer lower branch_id",
        "selected_output_shape": {
            "selected_branch_id": "...",
            "ranking": [
                {
                    "branch_id": "...",
                    "constraint_violation_count": 0,
                    "min_score": 0,
                    "mean_score": 0.0,
                    "branch_fidelity": 0,
                    "fixable_issue_count": 0,
                }
            ],
            "reason": "Computed from judge JSON, not by the LLM.",
        },
    }
    common["stages"] = build_staged_spec(entry)
    return common


def output_path_for(index: Any) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"{index}_request_{timestamp}.jsonl"


def find_entry_by_index(entries: list[dict[str, Any]], target_index: int) -> dict[str, Any]:
    for entry in entries:
        if entry.get("index") == target_index:
            return entry
    raise ValueError(f"TARGET_INDEX={target_index} was not found in {INPUT_PATH}")


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if BRANCH_COUNT < 1:
        raise ValueError("BRANCH_COUNT must be >= 1")

    entries = load_jsonl(INPUT_PATH)
    selected_entry = find_entry_by_index(entries, TARGET_INDEX)
    request = build_request(selected_entry)
    output = output_path_for(selected_entry.get("index"))
    write_jsonl([request], output)

    print(f"Wrote ToT staged request for index={TARGET_INDEX} to: {output}")


if __name__ == "__main__":
    main()
