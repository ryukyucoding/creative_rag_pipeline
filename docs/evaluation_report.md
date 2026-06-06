# Evaluation Report

## 1. Evaluation 設計

本次 evaluation 分為兩個面向：

**面向一：WritingBench Checklist（任務品質）**
使用 WritingBench 原生的 5 個 criterion，在 candidate 生成時由 LLM judge（Qwen2.5-14B-Instruct）對照 checklist 打 1–10 分。此步驟已整合在 `build_reranker_pool_hf.py` 的 judge stage，每個 candidate 生成後立即評分。最終從 `all_reranker_result` 的 `ranking` 陣列重建各 RAG variant 的 mean/min score，做 ablation 比較。

**面向二：創意維度（LLM-as-Judge）**
額外設計 5 個創意維度，由 LLM judge 在 reranking 結束後對最終輸出評分（1–10 分）：

| 維度 | 定義 |
|---|---|
| Novelty | 出乎意料但服務敘事的驚喜感 |
| Imagery Originality | 比喻與意象的新鮮「組合」 |
| Conceptual Originality | 核心概念的新穎程度 |
| Character Originality | 角色動機與性格組合的獨特性 |
| Worldbuilding Innovation | 世界規則的創新性與對情節的影響 |

**Baseline 設計**：使用 `none` variant（無 RAG）的三個 branch 各自評分後取平均，代表「純 LLM，無任何 RAG 介入」的創意基準線。

---

## 2. 實驗設置

| 項目 | 設定 |
|---|---|
| 測試集 | WritingBench Literature & Arts / English（96 題） |
| Generation model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Judge model | Qwen/Qwen2.5-14B-Instruct |
| Judge temperature | 0.0（deterministic） |
| RAG variants | none / random_association / near_relevance / creative_association |
| Branches per variant | 3（A / B / C） |
| Candidates per query | 12（4 variants × 3 branches） |
| RAG corpus | Gutenberg Poetry Corpus（~3M 詩句） |

---

## 3. 實驗結果

### WritingBench Ablation（各 variant 平均，across 所有 branch）

| Variant | mean\_score | min\_score | violation% |
|---|---|---|---|
| none（baseline） | 6.903 | 5.616 | 1.2% |
| random\_association | 6.873 | 5.508 | 0.8% |
| near\_relevance | 6.905 | 5.594 | 1.2% |
| **creative\_association** | **7.037** | **5.763** | **0.4%** |
| **selected（reranker 後）** | **7.421** | **6.219** | **1.0%** |

### 創意維度（selected pipeline vs none baseline）

| 維度 | Baseline (none avg) | Selected (RAG pipeline) | Δ |
|---|---|---|---|
| Novelty | 5.13 | 5.83 | +0.71 |
| Imagery Originality | 4.97 | 6.14 | **+1.16** |
| Conceptual Originality | 4.91 | 5.70 | +0.79 |
| Character Originality | 4.75 | 5.25 | +0.50 |
| Worldbuilding Innovation | 4.98 | 5.68 | +0.70 |
| **Mean** | **4.95** | **5.72** | **+0.77** |

96 題勝負分布：**81 題 RAG pipeline 勝出，12 題落後，3 題相同**（勝率 84%）。

---

## 4. 結果分析

**WritingBench 面向：** `creative_association` 在四個 variant 中 mean\_score 最高（7.037），比 none baseline 高 +0.13，且 constraint violation 率最低（0.4%）。Reranker 介入後分數進一步提升至 7.421，說明多候選選優機制有效。Reranker 最常選 `creative_association`（38/96 題，40%），其次為 `random_association`（33 題）與 `near_relevance`（24 題），`none` 僅被選中 1 次。

**創意維度面向：** 完整 pipeline 在全部 5 個維度均優於 baseline，整體提升 +0.77 分。提升最顯著的是 **Imagery Originality（+1.16）**，符合預期——Gutenberg Poetry corpus 以意象密集的詩句為主，RAG 素材天然對意象類創意有強化效果。落後的 12 題差值均小（最大 −1.1），而勝出題目差值最高達 +2.6，分布不對稱，整體明顯偏正。

**侷限性：**
- 創意評分來自同一 LLM judge（Qwen2.5-14B-Instruct），存在 self-evaluation bias 的潛在風險
- WritingBench checklist criteria 為 LLM 生成而非人工標注
- RAG corpus 偏重前 1920 年代英語詩歌，對現代風格的題目幫助可能有限
