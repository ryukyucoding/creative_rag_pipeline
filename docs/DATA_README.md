# Data Documentation

## 1. WritingBench (Test Query Set)

**Source**: https://github.com/X-PLUG/WritingBench  
**Paper**: Wu et al., "WritingBench: A Comprehensive Benchmark for Generative Writing", arXiv 2503.05244, NeurIPS 2025 Datasets and Benchmarks Track  
**License**: MIT  

### Schema

Each line in `benchmark_all.jsonl` (and the filtered test set) is a JSON object:

```json
{
  "index": 1,
  "domain1": "Literature & Arts",
  "domain2": "Poetry",
  "lang": "en",
  "query": "Write a poem about ...",
  "checklist": [
    {
      "name": "Criterion Name",
      "criteria_description": "What this criterion measures.",
      "1-2":  "Descriptor for score 1-2",
      "3-4":  "Descriptor for score 3-4",
      "5-6":  "Descriptor for score 5-6",
      "7-8":  "Descriptor for score 7-8",
      "9-10": "Descriptor for score 9-10"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `index` | int | Global query ID across all 1,000 entries |
| `domain1` | string | Top-level domain (e.g., "Literature & Arts") |
| `domain2` | string | Sub-domain (e.g., "Poetry", "Prose", "Novel Manuscript", "Screenplay") |
| `lang` | string | Language code (`"en"` or `"zh"`) |
| `query` | string | The creative writing prompt |
| `checklist` | array[5] | Five LLM-generated evaluation criteria with 5 score-band descriptors each |

**Filtered test set**: `domain1 == "Literature & Arts"` AND `lang == "en"` → ~96 queries.

---

## 2. Gutenberg Poetry Corpus (RAG Knowledge Base)

**Source**: http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz  
**Repository**: https://github.com/aparrish/gutenberg-poetry-corpus  
**Author**: Allison Parrish (2018)  
**License**: CC0 (public domain)  

### Schema

Each line in the `.ndjson.gz` file is a JSON object:

```json
{"s": "Shall I compare thee to a summer's day?", "gid": "1041"}
```

| Field | Type | Description |
|---|---|---|
| `s` | string | A single line of poetry text |
| `gid` | string | Project Gutenberg book ID the line was extracted from |

The corpus contains approximately 3 million lines drawn from ~3,000 public-domain poetry texts.

---

## 3. Design Rationale

### Why WritingBench over WritingPrompts (Reddit)?

- **Data freshness**: WritingBench was published in 2025; WritingPrompts data collected from Reddit up to ~2018 risks training-data contamination for modern LLMs.
- **Built-in evaluation**: Each query ships with 5 structured, rubric-style criteria (score bands 1–10), enabling automatic evaluation without a human panel.
- **Domain specificity**: The "Literature & Arts" subset covers Poetry, Prose, Novel Manuscript, and Screenplay — exactly the creative modalities targeted by this pipeline.

### Why Gutenberg Poetry over Wikipedia?

- **Self-contained units**: Each corpus line is a single poetic line — already a natural, self-contained imagery unit. No sentence-boundary chunking or sliding-window segmentation is required before indexing.
- **License**: CC0 — completely unrestricted use, no attribution required in derivative outputs.
- **Scale for mid-similarity retrieval**: ~3 million lines give enough density that a top-k retrieval (k = 5–20) reliably surfaces stylistically relevant but non-trivially similar lines, avoiding both "too obvious" and "completely irrelevant" retrievals.
- **Style diversity**: Spans hundreds of authors and centuries of English poetry.

---

## 4. Known Limitations

### WritingBench
- The five `checklist` criteria per query are **LLM-generated** (not human-annotated). They are generally coherent but can occasionally be redundant or inconsistently calibrated across queries.
- The benchmark is bilingual (EN + ZH); this project uses only the English subset. Cross-lingual transfer is not evaluated.

### Gutenberg Poetry Corpus
- **English-only**: No multilingual support.
- **Pre-1920s style bias**: Project Gutenberg's public-domain cutoff means the corpus skews heavily toward Victorian, Romantic, and earlier English poetry. Modern free verse and contemporary styles are underrepresented.
- **Noisy extraction**: Line boundaries were heuristically extracted from raw text files; some lines may be fragments, OCR artifacts, or page metadata.
- **No semantic metadata**: Lines carry only a book ID (`gid`), not author, title, or date — limiting provenance-based filtering.
