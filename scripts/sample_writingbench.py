"""Filter WritingBench to Literature & Arts / English entries and write them out."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Filter WritingBench to a specific domain and language."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "raw" / "writingbench_all.jsonl",
        help="Path to the full WritingBench JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "test_set" / "test_set_lit_arts_en.jsonl",
        help="Path for the filtered output JSONL file.",
    )
    parser.add_argument(
        "--domain",
        default="Literature & Arts",
        help="Value to match against the 'domain1' field.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Value to match against the 'lang' field.",
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
            f"Input file not found: {args.input}\n"
            "Run 'bash scripts/download_data.sh' first."
        )

    all_entries: list[dict] = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))

    filtered: list[dict] = [
        e for e in all_entries
        if e.get("domain1") == args.domain and e.get("lang") == args.lang
    ]

    print(f"Total entries in source : {len(all_entries)}")
    print(f"After filtering ({args.domain!r}, lang={args.lang!r}): {len(filtered)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for entry in filtered:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(filtered)} entries to: {args.output}")

    subdomain_counts: Counter[str] = Counter(e.get("domain2", "unknown") for e in filtered)
    print("\nSubdomain breakdown:")
    for subdomain, count in sorted(subdomain_counts.items(), key=lambda x: -x[1]):
        print(f"  {subdomain:<30} {count:>4}")


if __name__ == "__main__":
    main()
