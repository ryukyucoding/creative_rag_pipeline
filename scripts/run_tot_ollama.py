"""Run staged Tree-of-Thought requests with a local Ollama model."""

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from rerank_tot_candidats import rerank


DEFAULT_INPUT = Path("data/runs/llama31_8b_staged_tot_requests_lit_arts_en.jsonl")
DEFAULT_OUTPUT = Path("data/runs/llama31_8b_staged_tot_outputs_lit_arts_en.jsonl")
DEFAULT_GENERATION_MODEL = "llama3.1:8b"
DEFAULT_JUDGE_MODEL = "qwen2.5:14b-instruct-q4_K_M"
HF_TO_OLLAMA_MODEL = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": DEFAULT_GENERATION_MODEL,
}
DRAFT_FORMAT_INSTRUCTIONS = """

Draft output contract:
Return exactly one valid JSON object with these top-level keys only:
{
  "branch_id": "same id as the branch plan",
  "candidate_answer": "a complete, directly usable answer as one plain string",
  "self_check_notes": ["short note about constraints covered"]
}
Do not put outline headings or act numbers as JSON keys. Put the whole answer,
including headings and bullets if useful, inside candidate_answer as text.
The candidate_answer must be complete enough for judging, not a stub.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged ToT request specs against Ollama."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Ollama model tag for constraint mapping, branch expansion, drafting, and final revision.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Ollama model tag for LLM judging.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat endpoint.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--candidate-retries",
        type=int,
        default=1,
        help="Retry draft generation when a candidate is malformed or incomplete.",
    )
    parser.add_argument(
        "--min-candidate-chars",
        type=int,
        default=600,
        help="Minimum candidate_answer character count before judging.",
    )
    return parser.parse_args()


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


def resolve_model(request_model: Optional[str], cli_model: Optional[str]) -> str:
    if cli_model:
        return cli_model
    if request_model in HF_TO_OLLAMA_MODEL:
        return HF_TO_OLLAMA_MODEL[request_model]
    return request_model or DEFAULT_GENERATION_MODEL


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text[:500]}")
    return json.loads(match.group(1))


def ollama_chat(
    url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    num_ctx: int,
    timeout: int,
) -> Any:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it and pull the model first:\n"
            "  ollama pull llama3.1:8b\n"
            "  ollama serve"
        ) from exc

    body = json.loads(raw)
    content = body.get("message", {}).get("content", "")
    return extract_json(content)


def fill_template(template: str, replacements: dict[str, Any]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(
            "{" + key + "}",
            json.dumps(value, ensure_ascii=False, indent=2),
        )
    return rendered


def candidate_validation_errors(candidate: Any, min_chars: int) -> list[str]:
    if not isinstance(candidate, dict):
        return ["candidate is not a JSON object"]

    errors: list[str] = []
    allowed_keys = {"branch_id", "candidate_answer", "self_check_notes"}
    unexpected_keys = sorted(str(key) for key in candidate if key not in allowed_keys)
    if unexpected_keys:
        errors.append(
            "unexpected top-level keys: " + ", ".join(unexpected_keys[:5])
        )

    if "branch_id" not in candidate:
        errors.append("missing branch_id")
    if "self_check_notes" not in candidate:
        errors.append("missing self_check_notes")

    answer = candidate.get("candidate_answer")
    if not isinstance(answer, str):
        errors.append("candidate_answer must be one plain string")
        answer_text = json.dumps(answer, ensure_ascii=False) if answer is not None else ""
    else:
        answer_text = answer.strip()

    if len(answer_text) < min_chars:
        errors.append(
            f"candidate_answer too short: {len(answer_text)} chars < {min_chars}"
        )

    suspicious_keys = [
        str(key)
        for key in candidate
        if re.match(r"^-?\d|^-\d|^[-–]\d", str(key).strip())
    ]
    if suspicious_keys:
        errors.append(
            "malformed outline keys detected: " + ", ".join(suspicious_keys[:5])
        )

    return errors


def build_retry_instruction(errors: list[str]) -> str:
    return (
        "\n\nThe previous draft was rejected by the local quality gate for these reasons:\n"
        + json.dumps(errors, ensure_ascii=False, indent=2)
        + "\nRegenerate the candidate from scratch. Follow the draft output contract exactly."
    )


def same_branch_id(left: Any, right: Any) -> bool:
    return str(left) == str(right)


def find_by_branch_id(items: list[Any], branch_id: Any) -> Optional[Any]:
    for item in items:
        if isinstance(item, dict) and same_branch_id(item.get("branch_id") or item.get("id"), branch_id):
            return item
    return None


def run_record(
    record: dict[str, Any],
    generation_model: str,
    judge_model: str,
    ollama_url: str,
    temperature: float,
    judge_temperature: float,
    num_ctx: int,
    timeout: int,
    candidate_retries: int,
    min_candidate_chars: int,
) -> dict[str, Any]:
    if record.get("mode") != "staged":
        raise ValueError("run_tot_ollama.py only supports staged request specs.")

    stages = record["stages"]
    print(f"  - map constraints for index={record.get('index')}", flush=True)
    constraint_map = ollama_chat(
        ollama_url,
        generation_model,
        stages["constraint_map"]["system"],
        stages["constraint_map"]["user"],
        temperature,
        num_ctx,
        timeout,
    )

    print(f"  - expand branches for index={record.get('index')}", flush=True)
    expand_user = fill_template(
        stages["expand_branches"]["user_template"],
        {"constraint_map_json": constraint_map},
    )
    branch_expansion = ollama_chat(
        ollama_url,
        generation_model,
        stages["expand_branches"]["system"],
        expand_user,
        temperature,
        num_ctx,
        timeout,
    )

    branches = branch_expansion.get("branches", [])
    if not isinstance(branches, list) or not branches:
        raise ValueError(f"No branches returned for index={record.get('index')}")

    candidates: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    candidate_retry_notes: list[dict[str, Any]] = []
    selected_branches = branches[: int(record.get("branch_count", len(branches)))]
    print(
        f"  - got {len(branches)} branch plan(s); drafting {len(selected_branches)}",
        flush=True,
    )
    for branch_no, branch in enumerate(selected_branches, 1):
        branch_id = branch.get("id", branch_no) if isinstance(branch, dict) else branch_no
        print(
            f"  - draft branch {branch_id} ({branch_no}/{len(selected_branches)})",
            flush=True,
        )
        draft_user = (
            fill_template(
                stages["draft_branch"]["user_template"],
                {
                    "constraint_map_json": constraint_map,
                    "branch_plan_json": branch,
                },
            )
            + DRAFT_FORMAT_INSTRUCTIONS
        )
        candidate: Any = None
        errors: list[str] = []
        for attempt in range(candidate_retries + 1):
            if attempt > 0:
                print(
                    f"  - retry draft branch {branch_id} after validation failure",
                    flush=True,
                )
            candidate = ollama_chat(
                ollama_url,
                generation_model,
                stages["draft_branch"]["system"],
                draft_user if attempt == 0 else draft_user + build_retry_instruction(errors),
                temperature if attempt == 0 else min(temperature, 0.3),
                num_ctx,
                timeout,
            )
            errors = candidate_validation_errors(candidate, min_candidate_chars)
            if not errors:
                break

        if errors:
            print(
                f"  - branch {branch_id} kept with validation warnings: {errors}",
                flush=True,
            )
        candidate_retry_notes.append(
            {
                "branch_id": branch_id,
                "attempts": attempt + 1,
                "validation_errors": errors,
            }
        )
        candidates.append(candidate)

        print(f"  - judge branch {branch_id}", flush=True)
        judge_user = fill_template(
            stages["judge_branch"]["user_template"],
            {"candidate_json": candidate},
        )
        judgment = ollama_chat(
            ollama_url,
            judge_model,
            stages["judge_branch"]["system"],
            judge_user,
            judge_temperature,
            num_ctx,
            timeout,
        )
        judgments.append(judgment)

    print("  - rerank candidates", flush=True)
    rerank_result = rerank(judgments)
    selected_branch_id = rerank_result.get("selected_branch_id")
    selected_candidate = find_by_branch_id(candidates, selected_branch_id)
    selected_judgment = find_by_branch_id(judgments, selected_branch_id)
    selected_branch = find_by_branch_id(selected_branches, selected_branch_id)
    if selected_candidate is None or selected_judgment is None:
        raise ValueError(
            f"Could not resolve selected branch {selected_branch_id!r} for index={record.get('index')}"
        )

    print("  - revise selected candidate", flush=True)
    final_stage = stages["final_revision"]
    synth_user = fill_template(
        final_stage["user_template"],
        {
            "constraint_map_json": constraint_map,
            "rerank_result_json": rerank_result,
            "selected_branch_json": selected_branch,
            "selected_judgment_json": selected_judgment,
            "selected_candidate_json": selected_candidate,
        },
    )
    final = ollama_chat(
        ollama_url,
        generation_model,
        final_stage["system"],
        synth_user,
        temperature,
        num_ctx,
        timeout,
    )

    return {
        "index": record.get("index"),
        "generation_model": generation_model,
        "judge_model": judge_model,
        "branch_count": record.get("branch_count"),
        "constraint_map": constraint_map,
        "branch_expansion": branch_expansion,
        "candidates": candidates,
        "judgments": judgments,
        "candidate_retry_notes": candidate_retry_notes,
        "rerank_result": rerank_result,
        "selected_branch": selected_branch,
        "selected_candidate": selected_candidate,
        "selected_judgment": selected_judgment,
        "final": final,
    }


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)[args.start_at:]
    if args.max_items is not None:
        records = records[: args.max_items]

    if args.output.exists() and args.start_at == 0:
        args.output.unlink()

    for i, record in enumerate(records, 1):
        generation_model = resolve_model(record.get("model"), args.generation_model)
        judge_model = args.judge_model
        print(
            f"[{i}/{len(records)}] index={record.get('index')} "
            f"generation_model={generation_model} judge_model={judge_model}"
        )
        result = run_record(
            record,
            generation_model,
            judge_model,
            args.ollama_url,
            args.temperature,
            args.judge_temperature,
            args.num_ctx,
            args.timeout,
            args.candidate_retries,
            args.min_candidate_chars,
        )
        append_jsonl(args.output, result)

    print(f"Wrote {len(records)} staged ToT output(s) to: {args.output}")


if __name__ == "__main__":
    main()
