# Creative RAG Pipeline

這個專案是一個 training-free 的創意寫作增強流程。現在的主軸是 staged Tree-of-Thought：先把使用者題目整理成約束，再產生多個分支方案，分別寫候選答案，用 WritingBench checklist 評分，最後用程式 rerank 選出最佳候選並做 final revision。


## 快速開始

```bash
# 1. 下載原始資料：WritingBench + Gutenberg Poetry Corpus
bash scripts/download_data.sh

# 2. 篩出 Literature & Arts / English 測試集
python scripts/sample_writingbench.py

# 3. 檢查 Gutenberg Poetry Corpus 是否可讀
python scripts/inspect_gutenberg.py
```

已處理好的測試集在 `data/test_set/test_set_lit_arts_en.jsonl`。

## 資料概覽

| Dataset | 用途 | 大小 | 筆數 |
|---|---|---:|---:|
| WritingBench - Lit & Arts EN | 測試題目與 checklist | 約 1.4 MB | 約 96 題 |
| Gutenberg Poetry Corpus | 未來 RAG knowledge base | 約 52 MB 壓縮檔 | 約 300 萬行 |

## ToT 完整流程

目前流程以單題 staged request 為單位。`build_tot_requests.py` 只產生 request spec，不會 call model；真正 call Ollama API 的是 `run_tot_ollama.py`。

```text
WritingBench 題目
      ↓
build_tot_requests.py
      ↓
request JSONL
      ↓
run_tot_ollama.py
      ↓
constraint_map
      ↓
expand_branches
      ↓
K 個 branches
      ↓
draft_branch + judge_branch
      ↓
K 份 candidate + judgment
      ↓
programmatic rerank
      ↓
final_revision
      ↓
output JSONL
```



### API 呼叫次數

令 `K = BRANCH_COUNT`。在沒有 retry 的情況下，單題會 call Ollama API：

```text
1 次 constraint_map
+ 1 次 expand_branches
+ K 次 draft_branch
+ K 次 judge_branch
+ 1 次 final_revision
= 2K + 3 次 API call
```

例如 `BRANCH_COUNT = 3` 時，基礎呼叫數是 `2 * 3 + 3 = 9` 次。

額外呼叫來源：

- draft candidate validation 失敗時，每個 branch 最多額外 `CANDIDATE_RETRIES` 次 draft call。
- final revision validation 失敗時，最多額外 `FINAL_REVISION_RETRIES` 次 final call。
- programmatic rerank 不 call model，只用 judge JSON 做 deterministic sorting。

## 產生 Request

Script: `scripts/build_tot_requests.py`

改設定可改檔案前面的全域參數：

```python
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BRANCH_COUNT = 3
TARGET_INDEX = 180 
```
TARGET_INDEX指 `data/test_set/test_set_lit_arts_en.jsonl` 裡的哪個題目

固定 input：

```text
data/test_set/test_set_lit_arts_en.jsonl
```

執行：

```bash
python scripts/build_tot_requests.py
```

輸出會放在 `data/runs/`，檔名格式：

```text
{index}_request_{timestamp}.jsonl
```

例如：

```text
data/runs/180_request_20260529_235516.jsonl
```

## 執行 ToT

Script: `scripts/run_tot_ollama.py`

先準備 Ollama model：

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama serve
```

執行：

```bash
python scripts/run_tot_ollama.py
```

預設會自動讀取 `data/runs/` 裡最新的 `*_request_*.jsonl`。如果要指定特定 request 檔，就改 `run_tot_ollama.py` 前面的：

```python
REQUEST_PATH = None
```

輸出會放在 `data/runs/`，檔名格式：

```text
{index}_output_{timestamp}.jsonl
```

runner 目前的 model 分工：

- `GENERATION_MODEL = "llama3.1:8b"`：用於 constraint mapping、branch expansion、candidate drafting、final revision。
- `JUDGE_MODEL = "qwen2.5:14b-instruct-q4_K_M"`：用於 `judge_branch`。

runner 也會做 local validation：

- candidate 太短或 JSON shape 不對，會重試 draft。
- final answer 如果短於 selected candidate 的 70%，或看起來只是標題/摘要，會重試 final revision。
- retry notes 會寫入 output，方便追蹤。


## Prompt 與輸出細節

所有 stage 共用 system prompt：

```text
You are a creative-writing Tree-of-Thought controller.
Use visible, concise deliberation artifacts only: constraint maps, branch plans,
rubric scores, and revision notes. Do not reveal hidden chain-of-thought.
```

### 1. constraint_map

目的：只根據原始題目抽取約束，不看 WritingBench checklist。

輸出 schema：

```json
{
  "task_type": "...",
  "must_include": ["..."],
  "must_avoid": ["..."],
  "form_or_structure": ["..."],
  "style_or_tone": ["..."],
  "open_questions_or_ambiguities": ["..."],
  "likely_failure_modes": ["..."]
}
```

### 2. expand_branches

目的：產生 `K = BRANCH_COUNT` 個完整且彼此不同的創作路線。每個 branch 都必須能獨立完成整題，不可以把需求拆給不同 branch 分工。

輸出 schema：

```json
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
```

### 3. draft_branch

目的：針對單一 branch 寫出完整候選答案。候選答案要放在單一字串 `candidate_answer` 裡，不把章節或條列拆成 JSON key。

輸出 schema：

```json
{
  "branch_id": "A",
  "candidate_answer": "完整、可直接交給使用者的答案...",
  "self_check_notes": ["..."]
}
```

runner 會檢查：

- top-level keys 只能是 `branch_id`、`candidate_answer`、`self_check_notes`。
- `candidate_answer` 必須是字串。
- `candidate_answer` 長度至少是 `MIN_CANDIDATE_CHARS`。
- 如果不合格，最多重試 `CANDIDATE_RETRIES` 次。

### 4. judge_branch

目的：用 WritingBench checklist 評估每個 candidate。這是唯一會收到 checklist 的 stage。

`build_tot_requests.py` 的 `JUDGE_CHECKLIST_MODE` 預設是 `"compact"`，judge input 只放每個 criterion 的 `name` 和 `criteria_description`，降低 prompt 長度和 OOM 風險；需要完整 1-10 分數段描述時可改成 `"full"`。

模型只負責判斷，不負責計算 aggregate metrics。

輸出 schema：

```json
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
  "branch_fidelity": {
    "score_1_to_5": 0,
    "evidence": "...",
    "main_issue": "..."
  },
  "constraint_violations": [],
  "top_fix": "..."
}
```

### 5. programmatic rerank

`rerank_tot_candidats.py` 會從 judge JSON 推導 metrics：

```python
derived_metrics = {
    "mean_score": ...,
    "min_score": ...,
    "branch_fidelity": ...,
    "constraint_violation_count": ...,
    "fixable_issue_count": ...,
    "fixable_issues": [...],
}
```

排序規則：

1. `constraint_violation_count` 越少越好。
2. `min_score` 越高越好。
3. `mean_score` 越高越好。
4. `branch_fidelity` 越高越好。
5. `fixable_issue_count` 越少越好。
6. 最後用 `branch_id` 做 deterministic tie-breaker。

### 6. final_revision

目的：只修 selected candidate，不混用其他 candidate。final revision 的 input 是 compact payload：

```json
{
  "selected_candidate": {
    "branch_id": "A",
    "candidate_answer": "...",
    "self_check_notes": ["..."]
  },
  "selected_branch_plan": {
    "id": "A",
    "name": "..."
  },
  "selected_judgment": {
    "scores_by_criterion": ["..."],
    "branch_fidelity": {"...": "..."},
    "constraint_violations": [],
    "top_fix": "..."
  },
  "derived_metrics": {
    "mean_score": 0.0,
    "min_score": 0.0,
    "branch_fidelity": 0.0
  }
}
```

final prompt 的硬性要求：

- `final_answer` 必須是完整 revised answer，不可以只是標題。
- `final_answer` 必須至少和 `selected_candidate.candidate_answer` 一樣詳細。
- 不可以縮短 selected candidate。
- 不可以只是列修改建議。
- 必須把 `constraint_violations`、`top_fix`、每個 criterion 的 `main_issue`、`branch_fidelity.main_issue` 直接修進 `final_answer`。
- `revision_notes` 必須描述實際做了哪些改動。

輸出 schema：

```json
{
  "final_answer": "完整修訂後答案...",
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
```

runner 會檢查 final：

- `final_answer` 必須是非空字串。
- `final_answer` 長度必須至少是 selected candidate 的 `MIN_FINAL_CANDIDATE_RATIO`，目前是 70%。
- `final_answer <= 80 chars` 會被視為 title/summary，不合格。
- `revision_notes` 必須存在且非空。
- 如果不合格，最多重試 `FINAL_REVISION_RETRIES` 次。

## Output Record 欄位

`run_tot_ollama.py` 每題會輸出一筆 JSON object，主要欄位如下：

| 欄位 | 說明 |
|---|---|
| `constraint_map` | 題目約束整理 |
| `branch_expansion` | K 個 branch plans |
| `candidates` | 每個 branch 的候選答案 |
| `judgments` | 每個 candidate 的 judge JSON |
| `candidate_retry_notes` | candidate validation 與 retry 紀錄 |
| `rerank_result` | deterministic ranking 與 selected branch |
| `selected_branch` | 被選中的 branch plan |
| `selected_candidate` | 被選中的 candidate answer |
| `selected_judgment` | 被選中的完整 judge JSON |
| `selected_derived_metrics` | 程式推導出的 metrics |
| `final_revision_input` | 傳給 final revision 的 compact payload |
| `final_retry_notes` | final validation 與 retry 紀錄 |
| `final` | final answer、revision notes、rubric audit |

## 修改入口

| 想改的東西 | 位置 |
|---|---|
| 要跑哪一題、branch 數、request model id | `scripts/build_tot_requests.py` 頂部全域參數 |
| Ollama model、temperature、timeout、retry 次數 | `scripts/run_tot_ollama.py` 頂部全域參數 |
| 測試題目資料 | `data/test_set/test_set_lit_arts_en.jsonl` |
| 原始資料說明 | `docs/DATA_README.md` |

## RATT-inspired 設計說明

參考 RATT 的多分支探索概念，但針對 creative writing 任務進行簡化與改造。系統並不是直接針對 WritingBench query 生成單一答案，而是先從原始題目中抽取 constraint map，整理任務必須滿足的內容、格式、風格與可能失敗模式。接著，系統會展開多個 creative branches，每個 branch 都代表一條完整的敘事架構或創作路徑。

目前版本可以視為一個 lightweight / single-round 的 RATT-inspired multi-branch creative generation pipeline。它保留了 RATT 中「多路徑探索、分支評估、選擇較佳分支」的精神，未實作 RATT 的多輪迭代、節點整合與 retrieval-based correction。後續版本可以加入 Creative Association RAG，讓部分 branch 檢索中距離相關的靈感知識，用於提升文本的新穎性、隱喻性與創意轉換能力。

## 引用

```bibtex
@article{wu2025writingbench,
  title   = {WritingBench: A Comprehensive Benchmark for Generative Writing},
  author  = {Wu, Yuning and others},
  journal = {arXiv preprint arXiv:2503.05244},
  year    = {2025},
  note    = {NeurIPS 2025 Datasets and Benchmarks Track}
}
```

```text
Allison Parrish (2018). gutenberg-poetry-corpus.
https://github.com/aparrish/gutenberg-poetry-corpus
```
