"""LLM-as-judge creativity evaluation for none-variant (baseline) candidates.

Reads data/runs/all_data, scores all three none-variant branches per query on
the same five creativity dimensions as evaluation_creativity.py, then averages
the branch scores so each query produces one comparable row.

Usage:
    python scripts/evaluate_creativity_baseline.py
    python scripts/evaluate_creativity_baseline.py data/runs/all_data
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tot_hf import HFChatClient, JUDGE_MODEL_ID

RUNS_DIR = ROOT / "data" / "runs"
INPUT_PATH = RUNS_DIR / "all_data"

JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_NEW_TOKENS = 512

# Identical to evaluation_creativity.py so scores are directly comparable.
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

DIMS = [
    "novelty",
    "imagery_originality",
    "conceptual_originality",
    "character_originality",
    "worldbuilding_innovation",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / f"creativity_eval_baseline_{timestamp}.jsonl"


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


def evaluate_candidate(text: str, client: HFChatClient) -> Optional[dict[str, Any]]:
    user = USER_TEMPLATE.format(text=text[:6000])
    result = client.chat(
        JUDGE_MODEL_ID,
        SYSTEM_PROMPT,
        user,
        JUDGE_TEMPERATURE,
        JUDGE_MAX_NEW_TOKENS,
    )
    scores = {d: safe_dim(result, d) for d in DIMS}
    valid = [s for s in scores.values() if s["score"] > 0]
    scores["mean_score"] = round(sum(s["score"] for s in valid) / len(valid), 2) if valid else 0.0
    scores["raw_model_text"] = client.last_text[:1000]
    return scores


def average_branch_scores(branch_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Average per-dimension scores across branches. Skips parse-error entries."""
    averaged: dict[str, Any] = {}
    for dim in DIMS:
        valid = [b[dim]["score"] for b in branch_scores if b[dim]["score"] > 0]
        averaged[dim] = {
            "score": round(sum(valid) / len(valid), 2) if valid else 0,
            "reason": f"averaged over {len(valid)} branch(es)",
        }
    all_valid = [averaged[d]["score"] for d in DIMS if averaged[d]["score"] > 0]
    averaged["mean_score"] = round(sum(all_valid) / len(all_valid), 2) if all_valid else 0.0
    return averaged


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_PATH

    all_rows = load_jsonl(input_path)
    none_rows = [
        r for r in all_rows
        if r.get("pool_variant") == "none" and r.get("status") == "drafted"
    ]

    # Group by index so we can process all branches of a query together.
    by_index: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in none_rows:
        by_index[row.get("index")].append(row)

    out = output_path()
    client = HFChatClient()
    indices = sorted(by_index.keys())

    print(f"Evaluating {len(indices)} queries (none variant, all branches) → {out}", flush=True)
    for i, idx in enumerate(indices, 1):
        branches = by_index[idx]
        branch_scores = []

        for branch_row in branches:
            branch_id = branch_row.get("branch_id")
            text: str = branch_row.get("candidate", {}).get("candidate_answer", "")
            print(f"[{i}/{len(indices)}] index={idx} branch={branch_id}", flush=True)

            if not text.strip():
                scores: dict[str, Any] = {d: FALLBACK_SCORE for d in DIMS}
                scores["mean_score"] = 0.0
            else:
                scores = evaluate_candidate(text, client) or {d: FALLBACK_SCORE for d in DIMS}
            branch_scores.append(scores)

        row = {
            "index": idx,
            "pool_variant": "none",
            "branch_ids": [b.get("branch_id") for b in branches],
            "judge_model": JUDGE_MODEL_ID,
            "scores": average_branch_scores(branch_scores),
            "branch_scores": branch_scores,
        }
        append_jsonl(out, row)

    print(f"Done. Wrote {len(indices)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
