"""LLM-as-judge evaluation of creativity dimensions for selected candidates.

Reads data/runs/all_reranker_result_20260603_171146.jsonl, scores each
selected.candidate.candidate_answer on five creativity dimensions, and writes
results to a timestamped JSONL file under data/runs/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tot_hf import HFChatClient, JUDGE_MODEL_ID

RUNS_DIR = ROOT / "data" / "runs"
INPUT_PATH = RUNS_DIR / "all_reranker_result_20260603_171146.jsonl"

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = """\
You are an expert literary critic specialising in creative writing originality.
You will receive a piece of creative writing and evaluate it on exactly five dimensions.
Respond ONLY with a single valid JSON object — no markdown, no commentary outside the object.

JSON schema:
{
  "novelty": {
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  },
  "imagery_originality": {
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  },
  "conceptual_originality": {
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  },
  "character_originality": {
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  },
  "worldbuilding_innovation": {
    "score": <integer 1-10>,
    "reason": "<one sentence>"
  }
}

Scoring rubric (same for all dimensions):
  1-3  Commonplace, predictable, nothing surprising
  4-6  Some originality but largely familiar
  7-8  Genuinely fresh; stands out from typical works
  9-10 Exceptional; the kind of idea rarely seen

Dimension definitions:
- novelty: The piece surprises the reader in ways that still serve the story's
  purpose — unexpected yet fitting, counter-intuitive yet defensible on reflection.
- imagery_originality: Metaphors, images, and symbols form NEW combinations; the
  freshness lies in the pairing, not just the individual elements.
- conceptual_originality: The core premise or central idea of the piece is novel
  rather than a rehash of well-worn genre tropes.
- character_originality: Character motivations and personality combinations are
  uncommon; avoid characters that feel stock or archetypal without subversion.
- worldbuilding_innovation: The world's rules are inventive and demonstrably shape
  how society/individuals operate within the story; mere setting-dressing does not count.\
"""

USER_TEMPLATE = """\
Please evaluate the following creative writing excerpt on the five dimensions.

--- BEGIN EXCERPT ---
{text}
--- END EXCERPT ---

Return only the JSON object described in the system prompt.\
"""

FALLBACK_SCORE: dict[str, Any] = {
    "score": 0,
    "reason": "parse error — LLM did not return valid JSON",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"creativity_eval_{timestamp}.jsonl"


def safe_dim(result: Any, dim: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        return FALLBACK_SCORE
    entry = result.get(dim)
    if not isinstance(entry, dict):
        return FALLBACK_SCORE
    score = entry.get("score")
    if not isinstance(score, (int, float)) or not (1 <= score <= 10):
        return FALLBACK_SCORE
    return {"score": int(score), "reason": str(entry.get("reason", ""))}


def evaluate_candidate(
    text: str,
    client: HFChatClient,
) -> Optional[dict[str, Any]]:
    user = USER_TEMPLATE.format(text=text[:6000])
    result = client.chat(
        JUDGE_MODEL_ID,
        SYSTEM_PROMPT,
        user,
        JUDGE_TEMPERATURE,
        JUDGE_MAX_NEW_TOKENS,
    )
    dims = [
        "novelty",
        "imagery_originality",
        "conceptual_originality",
        "character_originality",
        "worldbuilding_innovation",
    ]
    scores = {d: safe_dim(result, d) for d in dims}
    valid = [s for s in scores.values() if s["score"] > 0]
    scores["mean_score"] = round(sum(s["score"] for s in valid) / len(valid), 2) if valid else 0.0
    scores["raw_model_text"] = client.last_text[:1000]
    return scores


def main() -> None:
    records = load_jsonl(INPUT_PATH)
    out = output_path()
    client = HFChatClient()

    print(f"Evaluating {len(records)} records → {out}", flush=True)
    for i, rec in enumerate(records, 1):
        idx = rec.get("index")
        text: str = rec.get("selected", {}).get("candidate", {}).get("candidate_answer", "")
        print(f"[{i}/{len(records)}] index={idx}", flush=True)

        if not text.strip():
            scores: dict[str, Any] = {"error": "empty candidate_answer"}
        else:
            scores = evaluate_candidate(text, client) or {"error": "evaluation failed"}

        row = {
            "index": idx,
            "pool_variant": rec.get("selected", {}).get("pool_variant"),
            "branch_id": rec.get("selected", {}).get("branch_id"),
            "judge_model": JUDGE_MODEL_ID,
            "scores": scores,
        }
        append_jsonl(out, row)

    print(f"Done. Wrote {len(records)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
