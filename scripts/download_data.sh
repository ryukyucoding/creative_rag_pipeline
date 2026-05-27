#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../data/raw"

mkdir -p "$RAW_DIR"

download_if_missing() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [[ -f "$dest" ]]; then
        echo "[skip] $label already exists at $dest"
    else
        echo "[download] $label ..."
        if command -v curl &>/dev/null; then
            curl -fSL --progress-bar -o "$dest" "$url"
        elif command -v wget &>/dev/null; then
            wget -q --show-progress -O "$dest" "$url"
        else
            echo "ERROR: neither curl nor wget found" >&2
            exit 1
        fi
        echo "[ok] $label downloaded"
    fi

    echo "[size] $(du -sh "$dest" | cut -f1)  $dest"
}

download_if_missing \
    "https://github.com/X-PLUG/WritingBench/raw/refs/heads/main/benchmark_query/benchmark_all.jsonl" \
    "$RAW_DIR/writingbench_all.jsonl" \
    "WritingBench (benchmark_all.jsonl)"

download_if_missing \
    "http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz" \
    "$RAW_DIR/gutenberg-poetry-v001.ndjson.gz" \
    "Gutenberg Poetry Corpus (gutenberg-poetry-v001.ndjson.gz)"

echo ""
echo "All downloads complete."
