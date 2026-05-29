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

## Current ToT Pipeline

The current Tree-of-Thought flow is staged and local-model oriented. It starts
from the processed WritingBench test set and writes all intermediate artifacts as
JSONL under `data/runs/`. Generation stages do not see the WritingBench
checklist; the full checklist is used only by the LLM judge and final audit.

### 1. Build ToT Request Specs

Script: `scripts/build_tot_requests.py`

Input:

- `data/test_set/test_set_lit_arts_en.jsonl`

Output:

- a request-spec JSONL file, for example `data/runs/tot_smoke_requests.jsonl`

Typical command:

```bash
python scripts/build_tot_requests.py \
  --preset llama31-8b-staged \
  --output data/runs/llama31_8b_staged_tot_requests_lit_arts_en.jsonl
```

What this creates per query:

- metadata: `index`, `query`, `branch_count`, `model`
- `stages.constraint_map`: extract query-only constraints
- `stages.expand_branches`: create K narrative-architecture branches
- `stages.draft_branch`: write one full candidate per branch
- `stages.judge_branch`: score each candidate with the full checklist
- `stages.final_revision`: revise only the selected best candidate
- `programmatic_rerank_policy`: deterministic ranking rule metadata

Useful knobs:

- `--max-items N`: small smoke run
- `--start-at N`: resume from an offset
- `--branch-count N`: number of branches
- `--mode single-call`: emit one all-in-one ToT prompt instead of staged specs

### 2. Run Staged ToT With Ollama

Script: `scripts/run_tot_ollama.py`

Input:

- staged request specs from step 1

Output:

- staged output JSONL, for example `data/runs/tot_smoke_outputs.jsonl`

Typical command:

```bash
ollama pull llama3.1:8b
ollama serve

python scripts/run_tot_ollama.py \
  --input data/runs/llama31_8b_staged_tot_requests_lit_arts_en.jsonl \
  --output data/runs/llama31_8b_staged_tot_outputs_lit_arts_en.jsonl
```

Runtime order for each query:

1. `constraint_map`: extract task type, must-include items, avoidances, form,
   tone, ambiguities, and likely failure modes from the query only.
2. `expand_branches`: generate K complete creative branches. Each branch is a
   different narrative architecture, not a different checklist dimension.
3. `draft_branch`: generate one complete candidate answer per branch.
4. branch-level RAG: planned, currently skipped.
5. `judge_branch`: use the full WritingBench checklist to score each candidate
   and check constraint violations.
6. programmatic rerank: sort by constraint violations, min score, then mean
   score, with deterministic tie-breakers.
7. `final_revision`: revise only the selected best candidate using its own judge
   feedback. Other candidates are not provided to this stage.

Each output record contains:

- `constraint_map`: query-only constraints
- `branch_expansion`: branch plans
- `candidates`: generated branch answers
- `judgments`: LLM judge scores and issues
- `candidate_retry_notes`: local validation/retry notes
- `rerank_result`: selected branch and ranking
- `selected_branch`, `selected_candidate`, `selected_judgment`: final inputs
- `final`: final answer plus rubric audit

### 3. Rerank Candidates

Script: `scripts/rerank_tot_candidats.py`

This is also imported directly by `run_tot_ollama.py`, so normally you do not
need to run it separately. It is useful when you already have judge outputs and
want to rerank them offline.

```bash
python scripts/rerank_tot_candidats.py \
  --input data/runs/judgments.jsonl \
  --output data/runs/rerank_results.jsonl
```

Ranking rule:

1. fewer `constraint_violations`
2. higher `min_score`
3. higher `mean_score`
4. fewer `fixable_issues` as a tie-breaker
5. lower/smaller `branch_id` as a final tie-breaker

## Where To Start Changing Things

- Change staged prompts or JSON output examples in `scripts/build_tot_requests.py`.
- Change local execution, retries, validation, Ollama model mapping, or output
  shape in `scripts/run_tot_ollama.py`.
- Change candidate selection logic in `scripts/rerank_tot_candidats.py`.
- Add branch-level RAG between drafting and judging when the CA-KB retrieval
  interface is ready.
- Change the test queries in `data/test_set/test_set_lit_arts_en.jsonl`.
- Existing example runs live in `data/runs/`, including index-180 and smoke
  request/output files.

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
