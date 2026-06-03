# Candidate Pool Reranking

`scripts/rerank_candidate_pool.py`

## 輸入 / 輸出

| | 路徑 |
|---|---|
| Input | `data/runs/{index}_reranker_pool_hf_{timestamp}.jsonl` |
| Output | `data/runs/{index}_reranker_result_{timestamp}.jsonl` |

每一行 input 是一個候選（一個 pool_variant × branch_id 的組合），由 `build_reranker_pool_hf.py` 產生。

---

## 排序流程

### Step 1 — 過濾非 drafted

只保留 `status == "drafted"` 的 row，其餘略過。

### Step 2 — Constraint violation 硬過濾

計算所有候選的 `constraint_violation_count` 最小值，只留下等於最小值的那些候選進入排序。

> 目的：不讓違規數多的候選與違規數少的混排。若全部候選都有違規，則選最少違規的那批，而不是直接放棄所有人。

### Step 3 — Composite Score 排序

對剩下的候選計算加權總分，分數越高排越前面。

#### 公式

```
score = 0.30 × (min_score / 10)
      + 0.25 × (mean_score / 10)
      + 0.20 × (branch_fidelity / 5)
      + 0.15 × (association_score / 5)
      − 0.10 × (fixable_issue_count / 10)
```

各指標先正規化到 \[0, 1\]，再乘以 weight，最後加總（fixable_issue 是 penalty，用減的）。

#### Weights

| 指標 | Weight | 原始 scale | 意義 |
|---|---|---|---|
| `min_score` | 0.30 | 1–10 | 最低單項分數，反映最弱的那個 criterion |
| `mean_score` | 0.25 | 1–10 | 所有 criterion 的平均分 |
| `branch_fidelity` | 0.20 | 1–5 | 是否忠實執行 branch 的寫作方向 |
| `association` | 0.15 | 1–5 | RAG 聯想素材的實際使用程度 |
| `fixable_issue_count` | 0.10 | count | 可修正問題數（penalty） |

> `min_score` weight 最高，是為了避免選出「平均高但有明顯短板」的候選。

---

## 輸出格式

```json
{
  "index": 180,
  "selected": {
    "pool_variant": "near_relevance",
    "branch_id": "A",
    "candidate": {},
    "judgment": {},
    "derived_metrics": {},
    "association_use": {}
  },
  "ranking": [
    {
      "rank": 1,
      "pool_variant": "near_relevance",
      "branch_id": "A",
      "score": 0.635,
      "reason": "violations=0, min=7.0, mean=8.2, fidelity=4.0, fixable=6, assoc=4.0, composite=0.635"
    }
  ]
}
```

- `selected`：排名第一的候選完整資料
- `ranking`：所有進入排序的候選（只含 min-violation 那批），附上各自的 composite score 與 reason 字串

---

## 調整 Weights

直接修改 `scripts/rerank_candidate_pool.py` 頂部的 `WEIGHTS` dict：

```python
WEIGHTS = {
    "min_score":       0.30,
    "mean_score":      0.25,
    "branch_fidelity": 0.20,
    "association":     0.15,
    "fixable_issue":   0.10,
}
```

所有 weight 不需要加總為 1，composite score 的絕對值會跟著變，但排序結果只取相對大小，不影響正確性。
