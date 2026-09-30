# Hybrid-RAG Final Latency Optimization Report

**Date:** September 30, 2026  
**Status:** Validated, Reconciled, and Permanently Applied to Production  
**Scope:** Multi-Document Hybrid-RAG Production Configuration Optimization  

---

## Executive Summary

Following comprehensive profiling and controlled experimentation across the 14 Indian Government welfare scheme corpus (82-query benchmark), the production retrieval configuration has been permanently optimized:

- **Original Baseline:** $K=15$ candidate pool, $V=3$ query perspectives, batch size 32
- **Optimized Configuration:** $K=10$ candidate pool, $V=3$ query perspectives, batch size 32
- **Production Config Values (`src/config.py`):**
  - `FUSION_TOP_K = 10`
  - `NUM_REFORMULATIONS = 3` ($V=3$: 1 original + 2 alternative perspectives)
  - `BATCH_SIZE = 32`

### Key Outcomes
- **Accuracy Preserved:** Hit@1 remains **97.56%** (80/82), Hit@3 remains **100.0%**, Hit@5 remains **100.0%**, MRR remains **0.9858**.
- **Significant Latency Reductions:**
  - Standardized Benchmark: **-23.8% Mean Latency** ($3.24\text{ s} \to 2.47\text{ s}$) and **-21.3% P95 Latency** ($3.71\text{ s} \to 2.92\text{ s}$).
  - End-to-End Multidoc Runner: **1.84 s Mean Latency**, **1.86 s Median Latency**, **2.15 s P95 Latency**.
- **Test Integrity:** 100% test pass rate (**38/38 tests passed**).

---

## 1. Original Baseline Configuration

The initial frozen production retrieval architecture established prior to optimization had the following parameters:

```python
# Baseline Configuration
RETRIEVAL_TOP_K              = 10   # Candidates retrieved per strategy per query variant
RRF_K                       = 60   # RRF ranking smoothing constant
FUSION_TOP_K                = 15   # Deduplicated candidate pool size passed to reranker (K=15)
FINAL_TOP_K                 = 5    # Final context chunks returned to reader
ENABLE_QUERY_REFORMULATION  = True
NUM_REFORMULATIONS          = 3    # Total query perspectives evaluated (V=3: original + 2 alternatives)
REFORMULATION_METHOD        = "linguistic"
ENABLE_MULTI_QUERY_RERANKING= True
BATCH_SIZE                  = 32   # CrossEncoder evaluation batch size
```

Under this baseline configuration, every user query generated:
- $V = 3$ query variants ($1$ original user query $+ 2$ diverse linguistic reformulations)
- $V \times 4 = 12$ retrieval passes across 3 FAISS index spaces (`Naive`, `Sentence-Window`, `Parent-Child`) + BM25 lexical search
- Deduplication and Canonical RRF fusion down to $K = 15$ top candidates
- $V \times K = 3 \times 15 = 45$ cross-encoder pairs scored per query

---

## 2. Optimized Production Configuration

The optimized production configuration permanently applied to [`src/config.py`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/src/config.py):

```python
# Optimized Production Configuration
RETRIEVAL_TOP_K              = 10   # Candidates retrieved per strategy per query variant
RRF_K                       = 60   # RRF ranking smoothing constant
FUSION_TOP_K                = 10   # Candidate pool size passed to reranker (Optimized: K=10)
FINAL_TOP_K                 = 5    # Final context chunks returned to reader
ENABLE_QUERY_REFORMULATION  = True
NUM_REFORMULATIONS          = 3    # Total query perspectives evaluated (V=3)
REFORMULATION_METHOD        = "linguistic"
ENABLE_MULTI_QUERY_RERANKING= True
BATCH_SIZE                  = 32   # CrossEncoder inference batch size
CROSS_ENCODER_BATCH_SIZE    = 32
```

In this optimized configuration, every user query now evaluates:
- $V = 3$ query variants
- Deduplication and Canonical RRF fusion down to $K = 10$ top candidates
- $V \times K = 3 \times 10 = \mathbf{30}$ cross-encoder pairs scored per query (a **33.3% reduction** in forward-pass compute load)

---

## 3. Why CrossEncoder Was Identified as the Bottleneck

Empirical profiling of the retrieval pipeline across the 14 government scheme documents isolated the exact latency contribution of each pipeline component:

| Pipeline Stage | Implementation | Mean Latency | % of Total Time |
| :--- | :--- | :--- | :--- |
| **Query Reformulation** | Rule-based heuristic perspective generator | 0.0003 s | 0.01% |
| **Dense Retrieval (FAISS)** | $3 \times 3$ Index searches (`all-MiniLM-L6-v2`) | 0.32 s | 9.9% |
| **Sparse Retrieval (BM25)** | 3 BM25 search passes over inverted index | 0.02 s | 0.6% |
| **Fusion & Deduplication** | Canonical Reciprocal Rank Fusion ($k=60$) | 0.002 s | 0.06% |
| **MMR Diversification** | Embedding cosine distance diversification | 0.01 s | 0.3% |
| **CrossEncoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` ($45$ pairs) | **2.37 s** | **76.5% – 85.0%** |
| **Total Query Time** | End-to-end `rag.query()` | **~3.10 – 3.24 s** | **100.0%** |

### Key Architectural Takeaways:
1. Dense vector and sparse BM25 lookups combined require $<0.35\text{ s}$ total.
2. CrossEncoder transformer forward passes accounted for over **76% of total query latency**.
3. Because $T_{\text{rerank}} \propto V \times K$, optimizing the candidate pool size $K$ without degrading recall directly minimizes reranker compute without modifying the deep neural network weights.

---

## 4. Controlled Experiments Summary

Controlled experiments were executed across 11 configurations using the 82-query benchmark:

### Experiment 1: Candidate Pool Size ($K$)
- Tested $K \in \{15, 12, 10, 8, 6\}$ with fixed $V=3, \text{batch}=32$.
- $K=15$: Hit@1 = 97.6%, MRR = 0.9858, Reranker Latency = 2.37 s.
- $K=12$: Hit@1 = 97.6%, MRR = 0.9858, Reranker Latency = 2.03 s (-14.3%).
- **$K=10$**: **Hit@1 = 97.6%**, **MRR = 0.9858**, **Reranker Latency = 1.72 s (-27.4%)**.
- $K=8$: Hit@1 = 97.6%, MRR = 0.9858, Reranker Latency = 1.48 s (-37.6%), but reduced safety margin for cross-document queries.
- $K=6$: Top-3 hit dropped to 98.8%, losing gold target documents.

### Experiment 2: Query Variants ($V$)
- Tested $V \in \{3, 2, 1\}$ with fixed $K=15, \text{batch}=32$.
- $V=2$: Hit@1 dropped to 96.34% (1 additional retrieval miss).
- $V=1$: Hit@1 dropped to 93.90% (3 additional retrieval misses).
- **Conclusion:** $V=3$ is mathematically required to maintain 97.56% Hit@1 accuracy across complex, multi-perspective phrasing.

### Experiment 3: Batch Size
- Tested $\text{batch} \in \{16, 32, 64\}$.
- Batch size 32 saturated multi-threaded CPU matrix operations without thread contention or memory paging thrash.

---

## 5. Baseline vs. Optimized Comparison Table

Using the standardized warm-up methodology established in [`reports/latency_baseline_reconciliation.md`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/reports/latency_baseline_reconciliation.md) across all 82 queries:

| Metric | Original Baseline ($K=15, V=3, \text{batch}=32$) | Optimized Configuration ($K=10, V=3, \text{batch}=32$) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Hit@1** | **97.56%** (80/82) | **97.56%** (80/82) | **0.00% (Identical)** |
| **Hit@3** | **100.00%** (82/82) | **100.00%** (82/82) | **0.00% (Identical)** |
| **Hit@5** | **100.00%** (82/82) | **100.00%** (82/82) | **0.00% (Identical)** |
| **MRR** | **0.9858** | **0.9858** | **0.0000 (Identical)** |
| **Mean Latency** | **3.24 s** | **2.47 s** | **-0.77 s (-23.8%)** |
| **Median Latency** | **3.07 s** | **2.33 s** | **-0.74 s (-24.1%)** |
| **P95 Latency** | **3.71 s** | **2.92 s** | **-0.79 s (-21.3%)** |
| **Pairs Scored** | 45 pairs | 30 pairs | **-15 pairs (-33.3%)** |

*(Note: In the full continuous run of `evaluate_multidoc.py`, warm steady-state latency achieved a mean of **1.84 s**, median of **1.86 s**, and P95 of **2.15 s**).*

---

## 6. Latency Reduction Percentages

$$\text{Mean Latency Reduction} = \frac{3.24 - 2.47}{3.24} \times 100 = \mathbf{23.77\%} \approx \mathbf{23.8\%}$$

$$\text{Median Latency Reduction} = \frac{3.07 - 2.33}{3.07} \times 100 = \mathbf{24.10\%} \approx \mathbf{24.1\%}$$

$$\text{P95 Latency Reduction} = \frac{3.71 - 2.92}{3.71} \times 100 = \mathbf{21.29\%} \approx \mathbf{21.3\%}$$

$$\text{CrossEncoder Compute Reduction} = \frac{45 - 30}{45} \times 100 = \mathbf{33.33\%} \approx \mathbf{33.3\%}$$

---

## 7. Retrieval Quality Comparison

### Overall Breakdown
- **Total Queries Evaluated:** 82
- **Hit@1:** 80/82 (97.56%)
- **Hit@3:** 82/82 (100.00%)
- **Hit@5:** 82/82 (100.00%)
- **Top-1 Misses:** Exactly 2 queries (Q09, Q71), identical to the original baseline.
- **Top-5 Failures:** 0 queries (100% recall of expected document schemes).

### Breakdown by Query Difficulty

| Difficulty Category | Query Count | Hit@1 | Hit@3 | Hit@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 23 | 100.0% | 100.0% | 100.0% | 1.0000 |
| **Medium** | 28 | 96.4% | 100.0% | 100.0% | 0.9762 |
| **Hard** | 19 | 100.0% | 100.0% | 100.0% | 1.0000 |
| **Cross-Document** | 12 | 91.7% | 100.0% | 100.0% | 0.9583 |

Both Top-1 misses (Q09 and Q71) were verified to have the target document in Rank 2/Rank 3 with zero coverage degradation.

---

## 8. Test Suite Results

The complete automated test suite was executed against the optimized production code:

```bash
python -m pytest tests/
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Sem VII\AI\Project\Implementation\Hybrid-RAG-Project
plugins: anyio-4.15.1
collected 38 items

tests\test_architecture.py .........                                     [ 23%]
tests\test_crossencoder_multiquery.py ........                           [ 44%]
tests\test_document_agnostic.py ...........                              [ 73%]
tests\test_query_reformulation.py ..........                             [100%]

============================= 38 passed in 40.05s =============================
```

- **Architecture Tests (`tests/test_architecture.py`):** 9/9 passed
- **Document-Agnostic Chunking Tests (`tests/test_document_agnostic.py`):** 11/11 passed
- **Multi-Query CrossEncoder Tests (`tests/test_crossencoder_multiquery.py`):** 8/8 passed
- **Query Reformulation Tests (`tests/test_query_reformulation.py`):** 10/10 passed
- **Total:** **38/38 passed (100% regression safety)**

---

## 9. Final Selected Production Configuration

The configuration is permanently set in [`src/config.py`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/src/config.py):

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `FUSION_TOP_K` | `10` | Cuts CrossEncoder pair evaluation by 33.3% while maintaining 100% Top-3 and Top-5 coverage. |
| `NUM_REFORMULATIONS` | `3` | Evaluates $V=3$ query perspectives (1 original + 2 linguistic perspectives) to prevent vocabulary mismatch. |
| `BATCH_SIZE` | `32` | Optimal tensor processing batch size for CrossEncoder inference on CPU. |
| `FINAL_TOP_K` | `5` | Highest-ranked contextual evidence passed to response generation. |
| `RRF_K` | `60` | Canonical reciprocal rank fusion constant. |
| `MMR_LAMBDA` | `0.7` | Optimal balance between relevance and non-redundancy. |

---

## 10. Reproducibility Information

### Restoring the Baseline
To reproduce the original baseline configuration, modify [`src/config.py`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/src/config.py):
```python
FUSION_TOP_K = 15  # Reverts to baseline 15 candidates
```
All code paths preserve backward compatibility.

### Running the Standardized Benchmark
```bash
python scripts/evaluate_multidoc.py
```
This automatically runs the single pre-warm query to eliminate PyTorch kernel cold-start artifacts, evaluates all 82 queries sequentially, and outputs JSON, CSV, and Markdown performance reports under `evaluation/results/` and `reports/`.

### Running Automated Regression Tests
```bash
python -m pytest tests/
```
Ensures 38/38 tests pass.
