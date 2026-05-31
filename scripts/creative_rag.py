"""Creative association retrieval for branch-level writing support.

This module intentionally starts with a transparent heuristic retriever instead
of an embedding index. The goal is to retrieve knowledge that can be transformed
into plot, character, setting, or imagery, not simply the nearest factual text.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = ROOT / "data" / "creative_kb" / "associations.jsonl"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "to",
    "with",
}


def load_kb(path: Path = DEFAULT_KB_PATH) -> list[dict[str, Any]]:
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


def tokenize(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False).lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", text)
        if token not in STOPWORDS
    }


def overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def association_score(
    query: str,
    constraint_map: dict[str, Any],
    branch_plan: dict[str, Any],
    item: dict[str, Any],
) -> float:
    """Score creative transfer potential for one KB item.

    The scoring is deliberately legible:
    - use_for overlap anchors the item to the task without requiring exact match.
    - mood overlap helps match tone.
    - branch overlap rewards items that can serve the selected branch.
    - exact fact overlap is lightly penalized to avoid ordinary nearest-neighbor
      factual retrieval dominating the creative association objective.
    """

    query_tokens = tokenize(query)
    constraint_tokens = tokenize(constraint_map)
    branch_tokens = tokenize(branch_plan)
    use_for_tokens = tokenize(item.get("use_for", []))
    mood_tokens = tokenize(item.get("mood", []))
    affordance_tokens = tokenize(item.get("creative_affordances", []))
    transfer_tokens = tokenize(item.get("transfer_targets", []))
    fact_tokens = tokenize(item.get("fact", ""))

    task_tokens = query_tokens | constraint_tokens
    creative_tokens = affordance_tokens | transfer_tokens | use_for_tokens

    score = 0.0
    score += 3.0 * overlap_score(task_tokens, use_for_tokens)
    score += 1.5 * overlap_score(constraint_tokens, mood_tokens)
    score += 2.0 * overlap_score(branch_tokens, creative_tokens)
    score += 1.0 * overlap_score(task_tokens, affordance_tokens)

    direct_fact_overlap = overlap_score(query_tokens, fact_tokens)
    if direct_fact_overlap > 0.35:
        score -= 1.0

    if score == 0.0:
        score = 0.05 * overlap_score(task_tokens | branch_tokens, creative_tokens)
    return score


def near_relevance_score(
    query: str,
    constraint_map: dict[str, Any],
    branch_plan: dict[str, Any],
    item: dict[str, Any],
) -> float:
    """Score ordinary relevance to the task and branch.

    This baseline intentionally rewards direct topical overlap more than
    creative transfer affordance. It is useful as a contrast against
    creative_association retrieval.
    """

    task_tokens = tokenize(query) | tokenize(constraint_map) | tokenize(branch_plan)
    item_tokens = tokenize(
        {
            "domain": item.get("domain", ""),
            "fact": item.get("fact", ""),
            "use_for": item.get("use_for", []),
            "mood": item.get("mood", []),
        }
    )
    return overlap_score(task_tokens, item_tokens)


def retrieve_associations(
    query: str,
    constraint_map: dict[str, Any],
    branch_plan: dict[str, Any],
    kb: list[dict[str, Any]],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    scored = [
        (association_score(query, constraint_map, branch_plan, item), item)
        for item in kb
    ]
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("id", ""))))
    return [
        {
            "score": round(score, 4),
            **item,
        }
        for score, item in scored[:top_k]
        if score > 0
    ]


def retrieve_near_relevance(
    query: str,
    constraint_map: dict[str, Any],
    branch_plan: dict[str, Any],
    kb: list[dict[str, Any]],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    scored = [
        (near_relevance_score(query, constraint_map, branch_plan, item), item)
        for item in kb
    ]
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("id", ""))))
    return [
        {
            "score": round(score, 4),
            **item,
        }
        for score, item in scored[:top_k]
        if score > 0
    ]


def retrieve_random_associations(
    kb: list[dict[str, Any]],
    top_k: int = 2,
    seed_material: str = "",
) -> list[dict[str, Any]]:
    rng = random.Random(seed_material)
    sampled = list(kb)
    rng.shuffle(sampled)
    return [
        {
            "score": None,
            **item,
        }
        for item in sampled[:top_k]
    ]


def build_association_packet(
    query: str,
    constraint_map: dict[str, Any],
    branch_plan: dict[str, Any],
    kb: list[dict[str, Any]],
    top_k: int = 2,
    mode: str = "creative_association",
) -> dict[str, Any]:
    branch_id = branch_plan.get("id") if isinstance(branch_plan, dict) else None
    if mode == "creative_association":
        items = retrieve_associations(query, constraint_map, branch_plan, kb, top_k)
        instruction = (
            "Use these as generative analogies. Transform them into plot, "
            "character behavior, setting logic, images, or emotional structure. "
            "Do not paste them as exposition unless the original task calls for it."
        )
    elif mode == "near_relevance":
        items = retrieve_near_relevance(query, constraint_map, branch_plan, kb, top_k)
        instruction = (
            "Use these topically relevant facts only when they can support the "
            "original task. Prefer clear relevance and coherence over surprise."
        )
    elif mode == "random_association":
        seed_material = json.dumps(
            {
                "query": query,
                "branch_id": branch_id,
                "branch_plan": branch_plan,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        items = retrieve_random_associations(kb, top_k, seed_material)
        instruction = (
            "This is a random-association baseline. Try to transform at least "
            "one item if it can be made coherent, but do not force irrelevant "
            "material into the answer."
        )
    elif mode == "none":
        items = []
        instruction = "No retrieval is used for this branch."
    else:
        raise ValueError(
            "mode must be one of: none, creative_association, near_relevance, "
            f"random_association; got {mode!r}"
        )
    return {
        "mode": mode,
        "branch_id": branch_id,
        "items": items,
        "instruction": instruction,
    }


def format_packet_for_prompt(packet: dict[str, Any]) -> str:
    return (
        "\n\nCreative association RAG packet:\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
        + "\nUse at least one association in a transformed way, while preserving all original constraints."
    )
