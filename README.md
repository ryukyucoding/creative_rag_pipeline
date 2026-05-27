# Creative RAG Pipeline

A training-free Creative Writing Enhancement Pipeline that combines multi-branch generation with Retrieval-Augmented Generation (RAG). The system retrieves stylistically relevant poetry lines from a large public-domain corpus to augment a language model's creative writing outputs, then evaluates results against structured criteria from the WritingBench benchmark.

## Quickstart

```bash
# 1. Download raw data (WritingBench ~14 MB + Gutenberg Poetry ~52 MB compressed)
bash scripts/download_data.sh

# 2. Filter WritingBench to Literature & Arts / English test set
python scripts/sample_writingbench.py

# 3. Verify the Gutenberg Poetry Corpus download
python scripts/inspect_gutenberg.py
```

Raw files land in `data/raw/` (gitignored). The processed test set is committed at `data/test_set/test_set_lit_arts_en.jsonl`.

## Data Overview

| Dataset | Role | Size | Entries |
|---|---|---|---|
| WritingBench — Lit & Arts EN | Test query set | ~1.4 MB | ~96 queries |
| Gutenberg Poetry Corpus | RAG knowledge base | ~52 MB (.gz) | ~3 M lines |

## Citations

```bibtex
@article{wu2025writingbench,
  title   = {WritingBench: A Comprehensive Benchmark for Generative Writing},
  author  = {Wu, Yuning and others},
  journal = {arXiv preprint arXiv:2503.05244},
  year    = {2025},
  note    = {NeurIPS 2025 Datasets and Benchmarks Track}
}
```

```
Allison Parrish (2018). gutenberg-poetry-corpus.
https://github.com/aparrish/gutenberg-poetry-corpus
```
