"""Sanity-check the Gutenberg Poetry Corpus download."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Inspect the Gutenberg Poetry Corpus ndjson.gz file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "raw" / "gutenberg-poetry-v001.ndjson.gz",
        help="Path to the gzipped ndjson corpus file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random sample lines to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {args.input}\n"
            "Run 'bash scripts/download_data.sh' first."
        )

    print(f"Reading {args.input} ...")

    all_lines: list[dict] = []
    gid_counter: Counter[str] = Counter()

    with gzip.open(args.input, "rt", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            all_lines.append(obj)
            gid_counter[obj.get("gid", "<missing>")] += 1

    total = len(all_lines)
    unique_books = len(gid_counter)

    print(f"\nTotal lines   : {total:,}")
    print(f"Unique book IDs (gid): {unique_books:,}")

    samples = random.sample(all_lines, min(args.samples, total))
    print(f"\n{args.samples} random sample lines:")
    for i, entry in enumerate(samples, 1):
        gid = entry.get("gid", "?")
        text = entry.get("s", "").strip()
        print(f"  [{i:>2}] gid={gid!r:>8}  {text!r}")

    print("\nTop-10 books by line count:")
    for gid, count in gid_counter.most_common(10):
        print(f"  gid={gid!r:<10}  {count:>6} lines")


if __name__ == "__main__":
    main()
