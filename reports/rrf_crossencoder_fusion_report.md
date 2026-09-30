# Controlled RRF + CrossEncoder Fusion Diagnostic Report

**Date:** 2026-09-29  
**Type:** Diagnostic Experiment & Feasibility Study  
**Scope:** Controlled evaluation of combining canonical RRF retrieval signals with CrossEncoder reranking scores.  
**Production Status:** DIAGNOSTIC ONLY. Zero production code modified.

---

## 1. Experiment Objective

In the Hybrid-RAG architecture, user queries flow through:
```
Query 
  → Multi-Query Reformulation 
  → Dense (Naive, Sentence-Window, Parent-Child) + Lexical (BM25) 
  → Canonical RRF Fusion 
  → MMR Diversification 
  → CrossEncoder Reranking
```

For the query `"Who is eligible for PMAY-U?"`, the multi-query retrieval and MMR pipeline successfully surfaces the `ELIGIBILITY` chunk into the candidate pool. However, CrossEncoder reranking evaluates `APPLICATION PROCESS` at Rank #1 and `ELIGIBILITY` at Rank #6.

The prior controlled ablation evaluated multi-query variant aggregation across the CrossEncoder, which elevated `ELIGIBILITY` to Rank #3. 

The objective of this controlled diagnostic experiment is to answer:
> **Can combining the upstream Reciprocal Rank Fusion (RRF) retrieval score with the downstream CrossEncoder score elevate `ELIGIBILITY` above `APPLICATION PROCESS`, and does this combination improve or degrade retrieval performance across the general benchmark?**

---

## 2. Exact Candidate-Pool Methodology

To ensure strict scientific control, the candidate pool was **not** retrieved independently per fusion strategy:

1. **Single Fixed Candidate Pool Per Query:**  
   The candidate pool entering the reranking stage (`res["diverse_results"]`) was retrieved **once** per query using `rag.query(query)`.
2. **Identical Candidates Across All Strategies:**  
   Every fusion strategy (CrossEncoder-only, RRF-only, all 11 weighted combinations, and rank-based reciprocal fusion) operated on the exact same ordered list of candidates.
3. **Preservation of Complete Lineage & Metadata:**  
   For every candidate in the candidate pool, the following fields were preserved without modification:
   - `document_id`
   - `source_file`
   - `chunk_id`
   - `source_chunk_id`
   - `retrieval_type`
   - `section`
   - `heading_path`
   - `rrf_score` (from upstream canonical RRF)
   - `retrieval_sources` (contributing dense/BM25 strategies)
   - `query_variants` (provenance across reformulated queries)
   - `text` / `retrieval_text` (canonical formatted text used for scoring)
4. **Strict Domain-Agnostic Design:**  
   Zero PMAY-specific rules, hardcoded section checks, or conditional bonuses were used.

---

## 3. Signal Normalization Methodology

CrossEncoder scores and RRF scores reside on fundamentally different scales and distributions:
- **CrossEncoder Logits:** Raw uncalibrated log-odds ranging from approximately `-11.5` to `+7.7`.
- **Canonical RRF Scores:** Reciprocal rank sums $\sum \frac{1}{k + r_i}$ ranging from approximately `0.014` to `0.195`.

Blindly summing these raw scores would cause CrossEncoder logits to completely overpower RRF scores (or vice-versa in extreme logit regimes). 

To ensure controlled combination, both signals were normalized to $[0.0, 1.0]$ within the candidate pool using **Min-Max Normalization**:

$$\text{norm\_CE}(c) = \frac{\text{raw\_CE}(c) - \min(\text{raw\_CE})}{\max(\text{raw\_CE}) - \min(\text{raw\_CE})}$$

$$\text{norm\_RRF}(c) = \frac{\text{raw\_RRF}(c) - \min(\text{raw\_RRF})}{\max(\text{raw\_RRF}) - \min(\text{raw\_RRF})}$$

- The highest candidate in a pool receives a normalized score of `1.0`.
- The lowest candidate receives `0.0`.
- Linear relative distance between candidates is strictly preserved.
- If all candidates share the same score, the normalized score defaults safely to `1.0`.

### Weighted Fusion Formulation
$$\text{final\_score}(c) = \alpha \cdot \text{norm\_CE}(c) + \beta \cdot \text{norm\_RRF}(c)$$

Eleven parameter combinations were tested: $\alpha \in [1.0, 0.9, \dots, 0.0]$ with $\beta = 1.0 - \alpha$.

### Rank-Based Reciprocal Fusion Formulation
In addition to score-based fusion, rank-based reciprocal fusion was evaluated using $k=60$ (matching the production canonical RRF constant):

$$\text{CE\_contrib}(c) = \frac{1}{60 + \text{rank}_{\text{CE}}(c)}, \quad \text{RRF\_contrib}(c) = \frac{1}{60 + \text{rank}_{\text{RRF}}(c)}$$

$$\text{rank\_fusion\_score}(c) = \text{CE\_contrib}(c) + \text{RRF\_contrib}(c)$$

---

## 4. Problematic Query Analysis: "Who is eligible for PMAY-U?"

### 4.1 Raw Candidate Pool (Retrieved Once, 14 Candidates)

| Pool # | Section | Source Chunk ID | Raw RRF | Norm RRF | Raw CE | Norm CE | RRF Rank | CE Rank |
|---|---|---|---|---|---|---|---|---|
| 1 | **APPLICATION PROCESS** | `...__naive__6` | **0.194471** | **1.0000** | **+7.2519** | **1.0000** | **#1** | **#1** |
| 2 | FREQUENTLY ASKED QUESTIONS | `...__naive__9` | 0.192548 | 0.9893 | +5.6814 | 0.7697 | #2 | #3 |
| 3 | FREQUENTLY ASKED QUESTIONS | `...__naive__8` | 0.169875 | 0.8635 | +2.1336 | 0.2494 | #3 | #9 |
| 4 | DETAILS | `...__naive__3` | 0.137758 | 0.6853 | +5.7328 | 0.7772 | #4 | #2 |
| 5 | DOCUMENTS REQUIRED | `...__naive__7` | 0.106335 | 0.5109 | +1.6337 | 0.1761 | #5 | #12 |
| 6 | **ELIGIBILITY** | `...__naive__5` | **0.091012** | **0.4258** | **+3.4143** | **0.4372** | **#6** | **#6** |
| 7 | BENEFITS | `...__naive__4` | 0.089887 | 0.4196 | +3.6637 | 0.4738 | #7 | #5 |
| 8 | Official PMAY-U Website | `...__naive__14` | 0.088254 | 0.4105 | +2.3208 | 0.2768 | #8 | #8 |
| 9 | FREQUENTLY ASKED QUESTIONS | `...__naive__10` | 0.076012 | 0.3426 | +5.0017 | 0.6700 | #9 | #4 |
| 10 | SOURCE | `...__naive__13` | 0.074886 | 0.3363 | +1.6607 | 0.1800 | #10 | #11 |
| 11 | FREQUENTLY ASKED QUESTIONS | `...__naive__12` | 0.057990 | 0.2426 | +0.9491 | 0.0757 | #11 | #13 |
| 12 | MINISTRY | `...__naive__1` | 0.014493 | 0.0011 | +2.5399 | 0.3090 | #12 | #7 |
| 13 | CATEGORIES | `...__naive__2` | 0.014286 | 0.0000 | +1.8555 | 0.2086 | #13 | #10 |
| 14 | FREQUENTLY ASKED QUESTIONS | `...__naive__11` | 0.014286 | 0.0000 | +0.4332 | 0.0000 | #14 | #14 |

### 4.2 Key Empirical Observation: Strict Dominance
Examining the two constituent signals for the two competing sections:
- **`APPLICATION PROCESS`**:
  - RRF Score: `0.194471` (Rank #1)
  - CE Score: `+7.2519` (Rank #1)
- **`ELIGIBILITY`**:
  - RRF Score: `0.091012` (Rank #6)
  - CE Score: `+3.4143` (Rank #6)

`APPLICATION PROCESS` **strictly dominates** `ELIGIBILITY` in both dimensions:
1. `raw_RRF(APPLICATION) > raw_RRF(ELIGIBILITY)`
2. `raw_CE(APPLICATION) > raw_CE(ELIGIBILITY)`
3. `norm_RRF(APPLICATION) = 1.0000 > norm_RRF(ELIGIBILITY) = 0.4258`
4. `norm_CE(APPLICATION) = 1.0000 > norm_CE(ELIGIBILITY) = 0.4372`

Because both normalized scores for `APPLICATION PROCESS` are strictly greater than those for `ELIGIBILITY`, **any linear combination $\alpha \cdot \text{CE} + \beta \cdot \text{RRF}$ with non-negative weights guarantees that `APPLICATION PROCESS` scores higher than `ELIGIBILITY`**.

---

## 5. Strategy Comparison Table (Problematic Query)

Query: *"Who is eligible for PMAY-U?"*

| Strategy | Application Rank | Eligibility Rank | Details Rank | Eligibility in Top-3? | Eligibility in Top-5? |
|---|---|---|---|---|---|
| **CrossEncoder-only (Baseline)** | **#1** | **#6** | #2 | NO | NO |
| **RRF-only (Baseline)** | **#1** | **#6** | #4 | NO | NO |
| Weighted ($\alpha=1.0, \beta=0.0$) | #1 | #6 | #2 | NO | NO |
| Weighted ($\alpha=0.9, \beta=0.1$) | #1 | #6 | #3 | NO | NO |
| Weighted ($\alpha=0.8, \beta=0.2$) | #1 | #6 | #3 | NO | NO |
| Weighted ($\alpha=0.7, \beta=0.3$) | #1 | #6 | #3 | NO | NO |
| Weighted ($\alpha=0.6, \beta=0.4$) | #1 | #7 | #3 | NO | NO |
| Weighted ($\alpha=0.5, \beta=0.5$) | #1 | #7 | #3 | NO | NO |
| Weighted ($\alpha=0.4, \beta=0.6$) | #1 | #7 | #3 | NO | NO |
| Weighted ($\alpha=0.3, \beta=0.7$) | #1 | #7 | #3 | NO | NO |
| Weighted ($\alpha=0.2, \beta=0.8$) | #1 | #7 | #4 | NO | NO |
| Weighted ($\alpha=0.1, \beta=0.9$) | #1 | #6 | #4 | NO | NO |
| Weighted ($\alpha=0.0, \beta=1.0$) | #1 | #6 | #4 | NO | NO |
| **Rank-based Fusion ($k=60$)** | **#1** | **#6** | **#3** | **NO** | **NO** |

> [!CAUTION]
> Notice that for balanced weights ($\alpha \in [0.3, 0.6]$), `ELIGIBILITY` actually **degrades** from Rank #6 to Rank #7 because `FAQ (naive__8)` has a high RRF score (Rank #3 in RRF) which pushes it past `ELIGIBILITY`.

---

## 6. Top 10 Candidates for Key Strategies

### A. CrossEncoder-only Baseline
| Rank | Section | Source Chunk ID | Final Score | Raw RRF | Raw CE | RRF Rk | CE Rk |
|---|---|---|---|---|---|---|---|
| 1 | **APPLICATION PROCESS** | `doc_...__naive__6` | 1.0000 | 0.194471 | +7.2519 | #1 | #1 |
| 2 | DETAILS | `doc_...__naive__3` | 0.7772 | 0.137758 | +5.7328 | #4 | #2 |
| 3 | FAQ | `doc_...__naive__9` | 0.7697 | 0.192548 | +5.6814 | #2 | #3 |
| 4 | FAQ | `doc_...__naive__10` | 0.6700 | 0.076012 | +5.0017 | #9 | #4 |
| 5 | BENEFITS | `doc_...__naive__4` | 0.4738 | 0.089887 | +3.6637 | #7 | #5 |
| 6 | **ELIGIBILITY** | `doc_...__naive__5` | 0.4372 | 0.091012 | +3.4143 | #6 | #6 |
| 7 | MINISTRY | `doc_...__naive__1` | 0.3090 | 0.014493 | +2.5399 | #12 | #7 |
| 8 | Official Website | `doc_...__naive__14` | 0.2768 | 0.088254 | +2.3208 | #8 | #8 |
| 9 | FAQ | `doc_...__naive__8` | 0.2494 | 0.169875 | +2.1336 | #3 | #9 |
| 10 | CATEGORIES | `doc_...__naive__2` | 0.2086 | 0.014286 | +1.8555 | #13 | #10 |

### B. RRF-only Baseline
| Rank | Section | Source Chunk ID | Final Score | Raw RRF | Raw CE | RRF Rk | CE Rk |
|---|---|---|---|---|---|---|---|
| 1 | **APPLICATION PROCESS** | `doc_...__naive__6` | 1.0000 | 0.194471 | +7.2519 | #1 | #1 |
| 2 | FAQ | `doc_...__naive__9` | 0.9893 | 0.192548 | +5.6814 | #2 | #3 |
| 3 | FAQ | `doc_...__naive__8` | 0.8635 | 0.169875 | +2.1336 | #3 | #9 |
| 4 | DETAILS | `doc_...__naive__3` | 0.6853 | 0.137758 | +5.7328 | #4 | #2 |
| 5 | DOCUMENTS REQUIRED | `doc_...__naive__7` | 0.5109 | 0.106335 | +1.6337 | #5 | #12 |
| 6 | **ELIGIBILITY** | `doc_...__naive__5` | 0.4258 | 0.091012 | +3.4143 | #6 | #6 |
| 7 | BENEFITS | `doc_...__naive__4` | 0.4196 | 0.089887 | +3.6637 | #7 | #5 |
| 8 | Official Website | `doc_...__naive__14` | 0.4105 | 0.088254 | +2.3208 | #8 | #8 |
| 9 | FAQ | `doc_...__naive__10` | 0.3426 | 0.076012 | +5.0017 | #9 | #4 |
| 10 | SOURCE | `doc_...__naive__13` | 0.3363 | 0.074886 | +1.6607 | #10 | #11 |

### C. Balanced Weighted Fusion ($\alpha=0.5, \beta=0.5$)
| Rank | Section | Source Chunk ID | Final Score | Raw RRF | Raw CE | RRF Rk | CE Rk |
|---|---|---|---|---|---|---|---|
| 1 | **APPLICATION PROCESS** | `doc_...__naive__6` | 1.0000 | 0.194471 | +7.2519 | #1 | #1 |
| 2 | FAQ | `doc_...__naive__9` | 0.8795 | 0.192548 | +5.6814 | #2 | #3 |
| 3 | DETAILS | `doc_...__naive__3` | 0.7312 | 0.137758 | +5.7328 | #4 | #2 |
| 4 | FAQ | `doc_...__naive__8` | 0.5564 | 0.169875 | +2.1336 | #3 | #9 |
| 5 | FAQ | `doc_...__naive__10` | 0.5063 | 0.076012 | +5.0017 | #9 | #4 |
| 6 | BENEFITS | `doc_...__naive__4` | 0.4467 | 0.089887 | +3.6637 | #7 | #5 |
| 7 | **ELIGIBILITY** | `doc_...__naive__5` | 0.4315 | 0.091012 | +3.4143 | #6 | #6 |
| 8 | Official Website | `doc_...__naive__14` | 0.3437 | 0.088254 | +2.3208 | #8 | #8 |
| 9 | DOCUMENTS REQUIRED | `doc_...__naive__7` | 0.3435 | 0.106335 | +1.6337 | #5 | #12 |
| 10 | SOURCE | `doc_...__naive__13` | 0.2582 | 0.074886 | +1.6607 | #10 | #11 |

### D. Rank-Based Fusion ($k=60$)
| Rank | Section | Source Chunk ID | Final Score | Raw RRF | Raw CE | RRF Rk | CE Rk |
|---|---|---|---|---|---|---|---|
| 1 | **APPLICATION PROCESS** | `doc_...__naive__6` | 0.032787 | 0.194471 | +7.2519 | #1 | #1 |
| 2 | FAQ | `doc_...__naive__9` | 0.032002 | 0.192548 | +5.6814 | #2 | #3 |
| 3 | DETAILS | `doc_...__naive__3` | 0.031754 | 0.137758 | +5.7328 | #4 | #2 |
| 4 | FAQ | `doc_...__naive__8` | 0.030366 | 0.169875 | +2.1336 | #3 | #9 |
| 5 | BENEFITS | `doc_...__naive__4` | 0.030310 | 0.089887 | +3.6637 | #7 | #5 |
| 6 | **ELIGIBILITY** | `doc_...__naive__5` | 0.030303 | 0.091012 | +3.4143 | #6 | #6 |
| 7 | FAQ | `doc_...__naive__10` | 0.030118 | 0.076012 | +5.0017 | #9 | #4 |
| 8 | Official Website | `doc_...__naive__14` | 0.029412 | 0.088254 | +2.3208 | #8 | #8 |
| 9 | DOCUMENTS REQUIRED | `doc_...__naive__7` | 0.029274 | 0.106335 | +1.6337 | #5 | #12 |
| 10 | MINISTRY | `doc_...__naive__1` | 0.028814 | 0.014493 | +2.5399 | #12 | #7 |

---

## 7. Full 5-Query Benchmark Evaluation

### 7.1 Per-Query Target Section Rank

| Query | Target Section | CE Baseline | RRF Baseline | $\alpha=0.9$ | $\alpha=0.7$ | $\alpha=0.5$ | $\alpha=0.3$ | $\alpha=0.2$ | $\alpha=0.1$ | $\alpha=0.0$ | Rank Fusion ($k=60$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1:** Who is eligible for PMAY-U? | ELIGIBILITY | #6 | #6 | #6 | #6 | #7 | #7 | #7 | #6 | #6 | **#6** |
| **Q2:** What are the eligibility criteria? | ELIGIBILITY | **#1** | #3 | **#1** | **#1** | **#1** | **#1** | #2 | #3 | #3 | **#1** |
| **Q3:** What benefits are available? | BENEFITS | **#1** | #2 | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | #2 | **#1** |
| **Q4:** How can beneficiaries apply? | APPLICATION PROCESS | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** |
| **Q5:** What documents are required? | DOCUMENTS REQUIRED | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** | **#1** |

---

## 8. Benchmark Metrics Summary & Regression Analysis

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Regressions vs. Baseline |
|---|:---:|:---:|:---:|:---:|---|
| **CrossEncoder-only (Baseline)** | **0.80** | **0.80** | **0.80** | **0.8333** | **None (0 regressions)** |
| Weighted ($\alpha=1.0, \beta=0.0$) | 0.80 | 0.80 | 0.80 | 0.8333 | None (0 regressions) |
| Weighted ($\alpha=0.9, \beta=0.1$) | 0.80 | 0.80 | 0.80 | 0.8333 | None (0 regressions) |
| Weighted ($\alpha=0.8, \beta=0.2$) | 0.80 | 0.80 | 0.80 | 0.8333 | None (0 regressions) |
| Weighted ($\alpha=0.7, \beta=0.3$) | 0.80 | 0.80 | 0.80 | 0.8333 | None (0 regressions) |
| Weighted ($\alpha=0.6, \beta=0.4$) | 0.80 | 0.80 | 0.80 | 0.8286 | Q1: #6 $\rightarrow$ #7 |
| Weighted ($\alpha=0.5, \beta=0.5$) | 0.80 | 0.80 | 0.80 | 0.8286 | Q1: #6 $\rightarrow$ #7 |
| Weighted ($\alpha=0.4, \beta=0.6$) | 0.80 | 0.80 | 0.80 | 0.8286 | Q1: #6 $\rightarrow$ #7 |
| Weighted ($\alpha=0.3, \beta=0.7$) | 0.80 | 0.80 | 0.80 | 0.8286 | Q1: #6 $\rightarrow$ #7 |
| Weighted ($\alpha=0.2, \beta=0.8$) | 0.60 | 0.80 | 0.80 | 0.7286 | Q1: #6 $\rightarrow$ #7, **Q2: #1 $\rightarrow$ #2** |
| Weighted ($\alpha=0.1, \beta=0.9$) | 0.60 | 0.80 | 0.80 | 0.7000 | **Q2: #1 $\rightarrow$ #3** |
| Weighted ($\alpha=0.0, \beta=1.0$) | 0.40 | 0.80 | 0.80 | 0.6000 | **Q2: #1 $\rightarrow$ #3, Q3: #1 $\rightarrow$ #2** |
| **RRF-only (Baseline)** | 0.40 | 0.80 | 0.80 | 0.6000 | **Q2: #1 $\rightarrow$ #3, Q3: #1 $\rightarrow$ #2** |
| **Rank-based Fusion ($k=60$)** | **0.80** | **0.80** | **0.80** | **0.8333** | **None (0 regressions)** |

### Regression Analysis Details:
1. **At $\alpha \ge 0.7$ (Low RRF weight):** Fusion acts as a noisy identity transform over CrossEncoder scores. The metrics remain static at `Hit@1 = 0.80`, `MRR = 0.8333`. There are zero gains and zero regressions.
2. **At $\alpha \in [0.3, 0.6]$ (Moderate RRF weight):** RRF injects noise into Query 1, dropping `ELIGIBILITY` from Rank #6 to #7, slightly reducing MRR from `0.8333` to `0.8286`.
3. **At $\alpha \le 0.2$ (High RRF weight):** Severe regressions occur on benchmark queries where CrossEncoder is strictly superior:
   - On Q2 (*"What are the eligibility criteria?"*), CrossEncoder achieves Rank #1, but RRF has it at Rank #3. High RRF weight regresses Q2 to Rank #2 or #3.
   - On Q3 (*"What benefits are available?"*), CrossEncoder achieves Rank #1, but RRF has it at Rank #2. High RRF weight regresses Q3 to Rank #2.
   - Hit@1 collapses from `80%` down to `40%`.

---

## 9. Latency Profile

| Stage | Latency | Complexity |
|---|---|---|
| **Retrieval Pipeline** (Multi-Query Dense + BM25 + Canonical RRF + MMR) | 1,699.8 ms | $\mathcal{O}(V \cdot (D + B) + N \log N)$ |
| **CrossEncoder Inference** (Candidate Pool of ~14 chunks) | 1,353.1 ms | $\mathcal{O}(N \cdot L_{\text{pair}})$ |
| **Fusion Computation** (Min-Max Normalization + Weighted Combination) | < 0.05 ms | $\mathcal{O}(N)$ |
| **Total Query Latency** | **3,052.9 ms** | Real-time interactive CPU |

The mathematical computation of fusion itself is negligible (<0.05 ms). However, the architectural utility of a computation depends entirely on whether it produces a better ranking.

---

## 10. Direct Answers to the 6 Core Diagnostic Questions

### 1. Does RRF contain useful information that CrossEncoder misses?
**NO.** In the candidate pool for `"Who is eligible for PMAY-U?"`, RRF does not contain any discriminative signal that favors `ELIGIBILITY` over `APPLICATION PROCESS`. In fact, both RRF and CrossEncoder place `APPLICATION PROCESS` at Rank #1 and `ELIGIBILITY` at Rank #6. 

Why? Because the dense retrievers and BM25 already over-matched the word *"eligible"* and *"PMAY-U"* against the large `APPLICATION PROCESS` chunk during initial retrieval, giving it top ranks across multiple retrieval lists. RRF simply accumulated those retrieval votes. Therefore, RRF shares the exact same bias as CrossEncoder for this query.

### 2. Does combining RRF + CrossEncoder move ELIGIBILITY above APPLICATION PROCESS?
**NO.** Under all 11 weighted combinations ($\alpha=0.0$ to $1.0$) and rank-based reciprocal fusion, `APPLICATION PROCESS` remains at Rank #1 and `ELIGIBILITY` remains at Rank #6 (or drops to #7). 

Because `APPLICATION PROCESS` strictly dominates `ELIGIBILITY` in both constituent dimensions ($\text{CE: } +7.25 \text{ vs } +3.41$; $\text{RRF: } 0.194 \text{ vs } 0.091$), no non-negative linear combination or monotonic rank aggregation can ever invert their ranking.

### 3. Which fusion weights, if any, improve Hit@1?
**NONE.** No weight configuration improves Hit@1 beyond the CrossEncoder baseline of `0.80` (`4/5`). In fact, as RRF weight $\beta$ increases above `0.7`, Hit@1 drops significantly:
- $\beta \le 0.7$: Hit@1 = `0.80` (`4/5`)
- $\beta = 0.8$: Hit@1 = `0.60` (`3/5`)
- $\beta = 1.0$: Hit@1 = `0.40` (`2/5`)

### 4. Does the improvement generalize to the other benchmark queries?
There is **no improvement** on the problematic query, and assigning meaningful weight to RRF actively **harms** general benchmark queries:
- For Q2 (*"What are the eligibility criteria?"*), CrossEncoder achieves Rank #1, whereas RRF ranks it #3. Fusing RRF causes a direct regression.
- For Q3 (*"What benefits are available?"*), CrossEncoder achieves Rank #1, whereas RRF ranks it #2. Fusing RRF causes a direct regression.

### 5. Does rank-based fusion behave differently from score-based fusion?
**Slightly, but inconsequentially.** Rank-based fusion ($k=60$) acts as a damped combination because $k=60$ is large relative to a candidate pool size of $14$. It successfully prevents the regressions seen under high RRF weights (maintaining Hit@1 at `0.80` and MRR at `0.8333`), but like weighted score fusion, it leaves `APPLICATION PROCESS` at Rank #1 and `ELIGIBILITY` at Rank #6.

### 6. Is there evidence that fusion should become part of the production architecture?
**NO.** There is definitive empirical and theoretical evidence **AGAINST** introducing RRF-CrossEncoder fusion into production:
1. **Mathematical Ineffectiveness:** When both scorers agree that a distractor chunk is Rank #1, post-retrieval fusion is mathematically incapable of fixing the ranking.
2. **Regression Risk:** Adding RRF scores directly degrades precision on queries where CrossEncoder is already optimal.
3. **Contrast with Query-Perspective Aggregation:** The prior ablation proved that **multi-query CrossEncoder aggregation** elevates `ELIGIBILITY` from Rank #6 to Rank #3 because reformulated queries provide genuine semantic discrimination that CrossEncoder can evaluate. RRF provides no such semantic discrimination.

---

## 11. Architectural Conclusion & Recommendations

| Strategy Evaluated | Problem Query Rank (Elig vs App) | Benchmark Hit@1 | Benchmark MRR | Production Readiness |
|---|---|:---:|:---:|:---:|
| **CrossEncoder-only (Current)** | App #1, Elig #6 | **0.80** | **0.8333** | **Current Production** |
| **RRF + CrossEncoder Fusion** | App #1, Elig #6 (or #7) | 0.40 – 0.80 | 0.6000 – 0.8333 | **REJECTED** |
| **Multi-Query CE Aggregation** *(prior)* | App #1, Elig #3 | 0.80 | 0.8667 | **Strongest Candidate** |

### Final Recommendation:
**DO NOT implement RRF + CrossEncoder score/rank fusion in production.**  
The diagnostic results confirm that RRF does not provide the missing signal needed to solve the `ELIGIBILITY` vs `APPLICATION PROCESS` ambiguity. Future efforts should remain focused on query-perspective reformulation and semantic reranking enhancements rather than hybrid score blending.
