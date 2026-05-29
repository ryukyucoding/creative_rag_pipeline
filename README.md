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
JSONL under `data/runs/`. The WritingBench checklist is included only in the
`judge_branch` prompt. Constraint mapping, branch expansion, drafting, and final
revision do not receive checklist data.

### 1. Build ToT Request Specs

Script: `scripts/build_tot_requests.py`

Input:

- `data/test_set/test_set_lit_arts_en.jsonl`

Output:

- a staged request-spec JSONL file, for example `data/runs/tot_smoke_requests.jsonl`

Typical command:

```bash
python scripts/build_tot_requests.py \
  --preset llama31-8b-staged \
  --output data/runs/llama31_8b_staged_tot_requests_lit_arts_en.jsonl
```

Useful knobs:

- `--max-items N`: small smoke run
- `--start-at N`: resume from an offset
- `--branch-count N`: number of branches

### 2. Run Staged ToT With Ollama

Script: `scripts/run_tot_ollama.py`

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:14b-instruct
ollama serve

python scripts/run_tot_ollama.py \
  --input data/runs/llama31_8b_staged_tot_requests_lit_arts_en.jsonl \
  --output data/runs/llama31_8b_staged_tot_outputs_lit_arts_en.jsonl
```

Model split:

- generation model: `llama3.1:8b` for constraint mapping, branch expansion, drafting, and final revision
- judge model: `qwen2.5:14b-instruct` for `judge_branch`

Runtime order for each query:

1. `constraint_map`: query-only constraint extraction.
2. `expand_branches`: generate K narrative-architecture branches.
3. `draft_branch`: generate one complete candidate answer per branch.
4. branch-level RAG: planned, currently skipped.
5. `judge_branch`: use the full WritingBench checklist to score candidates.
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
- `final`: final answer plus revision notes/audit

### Stage Prompts

All stages use this system prompt:

```text
You are a creative-writing Tree-of-Thought controller.
Use visible, concise deliberation artifacts only: constraint maps, branch plans,
rubric scores, and revision notes. Do not reveal hidden chain-of-thought.
```

`constraint_map` user prompt:

```text
Extract a concise constraint map from the user's creative-writing query only.
Separate explicit requirements from avoidances, style, structure, ambiguities,
and likely failure modes. Return exactly one JSON object matching this example
shape:
{
  "task_type": "...",
  "must_include": ["..."],
  "must_avoid": ["..."],
  "form_or_structure": ["..."],
  "style_or_tone": ["..."],
  "open_questions_or_ambiguities": ["..."],
  "likely_failure_modes": ["..."]
}

Input:
{generation_payload_json}
```

`expand_branches` user prompt template:

```text
Create exactly {branch_count} complete creative branches.
Each branch must be a complete solution path for the original task. Each branch
must cover every item in constraint_map.must_include, and constraint_coverage
must name each required item with a concrete plan for satisfying it. Do not split
required elements across separate branches: every branch must handle the whole
task. The branches must be genuinely different narrative architectures, not
minor variations or different evaluation dimensions. Do not score branches. Keep
each branch compact but complete enough to guide drafting.

Return exactly one JSON object matching this example shape:
{
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
          "plan": "..."
        }
      ],
      "distinctive_material": "...",
      "risk": "..."
    }
  ]
}

Original input:
{generation_payload_json}

Constraint map:
{constraint_map_json}
```

`draft_branch` user prompt template:

```text
Write one candidate answer for the selected branch.
Respect the original user query and the constraint map. Visibly execute the
selected branch's narrative_architecture. Do not fall back to a generic outline
or generic answer pattern. Use the branch's distinctive_material and follow its
development_path as the organizing logic for the candidate.

Return exactly one JSON object with only these keys:
branch_id, candidate_answer, self_check_notes.
candidate_answer must be a complete, directly usable answer as one plain string.
Do not use outline headings or act numbers as JSON keys; put all headings and
bullets inside candidate_answer.

Original input:
{generation_payload_json}

Constraint map:
{constraint_map_json}

Branch plan:
{branch_plan_json}
```

`judge_branch` user prompt template. This is the only stage that receives the
full WritingBench checklist:

```text
Use a simple rubric judge for one candidate answer.
Use the full WritingBench checklist below to score each criterion from 1 to 10.
Also check constraint violations against the original query requirements. Keep
the critique short and concrete.

Return exactly one JSON object matching this example shape:
{
  "branch_id": "A",
  "scores_by_criterion": [
    {
      "name": "...",
      "score_1_to_10": 0,
      "evidence": "...",
      "main_issue": "..."
    }
  ],
  "mean_score": 0.0,
  "min_score": 0,
  "constraint_violations": ["..."],
  "strengths": ["..."],
  "fixable_issues": ["..."]
}

Original input:
{judge_payload_json_with_full_checklist}

Candidate:
{candidate_json}
```

`final_revision` user prompt template:

```text
Revise only the selected best candidate into the final answer.
Use the selected candidate and its judge feedback to fix weaknesses. Do not use,
quote, blend, or borrow from any other candidate.

Return exactly one JSON object matching this example shape:
{
  "final_answer": "...",
  "revision_notes": ["..."],
  "rubric_audit": [
    {
      "criterion": "...",
      "score_1_to_10": 0,
      "evidence": "...",
      "remaining_risk": "..."
    }
  ]
}
The final_answer must be directly usable by the original user.

Original input:
{generation_payload_json}

Constraint map:
{constraint_map_json}

Programmatic rerank result:
{rerank_result_json}

Selected branch plan:
{selected_branch_json}

Selected judge feedback:
{selected_judgment_json}

Selected candidate:
{selected_candidate_json}
```

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
