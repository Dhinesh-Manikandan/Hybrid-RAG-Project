# Hybrid RAG: Multi-Representation Retrieval, Fusion & Grounded Generation

> **Branch Notice (`feature/05_hybrid_fusion_and_reranking`)**  
> This branch hosts the complete, production-ready implementation of the Hybrid RAG system — integrating multi-representation chunking, dual-channel hybrid retrieval, multi-query Cross-Encoder reranking, grounded LLM answer generation with provenance attribution, and a full Streamlit research dashboard. **This branch is currently maintained independently from `main`. If you clone this repository, make sure you are checked out on this branch.**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Quick Start: Clone & Run](#quick-start-clone--run)
  - [1. Clone Repository & Checkout Branch](#1-clone-repository--checkout-branch)
  - [2. Environment Setup](#2-environment-setup)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Ingest Documents & Build Indexes](#4-ingest-documents--build-indexes)
  - [5. Launch the Streamlit Web Application](#5-launch-the-streamlit-web-application)
- [Usage Modes](#usage-modes)
  - [Streamlit Research Web UI](#streamlit-research-web-ui)
  - [Python API / Programmatic Usage](#python-api--programmatic-usage)
  - [CLI Query Mode](#cli-query-mode)
- [Adding Custom Documents (Zero-Code Ingestion)](#adding-custom-documents-zero-code-ingestion)
- [Evaluation & Empirical Benchmarks](#evaluation--empirical-benchmarks)
  - [Retrieval Accuracy Metrics](#retrieval-accuracy-metrics)
  - [Running Evaluation Scripts](#running-evaluation-scripts)
  - [Running Automated Tests](#running-automated-tests)
- [Configuration Reference (`src/config.py`)](#configuration-reference-srcconfigpy)
- [Repository Structure](#repository-structure)
- [Preloaded Schemes](#preloaded-schemes)
- [License & Citation](#license--citation)

---

## Overview

Traditional RAG systems typically suffer from three major points of failure:
1. **Chunk Boundary Dilution:** Fixed-size chunking either splits essential context across chunks or pulls in irrelevant noise.
2. **Lexical vs. Semantic Mismatch:** Pure vector search misses exact identifiers (scheme acronyms, eligibility criteria numbers, legal sections), while pure keyword search fails at conceptual understanding.
3. **Hallucination & Provenance Loss:** Generative models fabricate answers when retrieved contexts are noisy or when user queries are out-of-scope.

**Hybrid-RAG** solves these challenges using a production-grade, multi-stage pipeline:
- **Multi-Representation Ingestion:** Simultaneously indexes documents across three structural representations: Standard Chunks, Sentence-Window chunks with expanded retrieval windows, and Parent-Child hierarchical chunks.
- **Dual-Channel Hybrid Search:** Combines dense semantic vector retrieval (FAISS + `all-MiniLM-L6-v2`) with sparse lexical retrieval (BM25).
- **Multi-Query Perspective Reformulation:** Generates $V=3$ domain-agnostic query variants (original, structural expansion, and lexical rephrasing) to capture diverse passage phrasings.
- **Canonical Reciprocal Rank Fusion (RRF) & MMR:** Normalizes and merges candidate pools across all representations while preventing duplicate chunk redundancy.
- **Multi-Query Cross-Encoder Reranking:** Computes joint query-passage relevance with `cross-encoder/ms-marco-MiniLM-L-6-v2` and aggregates multi-query scores.
- **Relevance Gate:** Evaluates top passage logit scores against a calibrated threshold (`-10.5`) to cleanly reject out-of-domain queries without hallucinating.
- **Strict Grounded Generation:** Generates concise answers using `Qwen/Qwen3-0.6B` / `Qwen3-8B` in non-thinking mode with direct source citations and metadata tracing.

---

## System Architecture

```text
                                 [ User Query ]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     [ Query Reformulation (V=3) ]              [ Direct Query ]
      - Original Query                           - Fallback Channel
      - Structural / Legal Expansion
      - Lexical Synonym Expansion
                    │
                    ├─────────────────────────────────────────────────┐
                    ▼                                                 ▼
        ┌──────────────────────┐                           ┌─────────────────────┐
        │ Dense Vector Search  │                           │ Sparse Lexical (BM25)│
        │     (FAISS HNSW)     │                           │    Token Inverted   │
        └──────────┬───────────┘                           └──────────┬──────────┘
                   │                                                  │
       ┌───────────┼──────────────┐                                   │
       ▼           ▼              ▼                                   ▼
  [ Naive ]  [ Sentence-   [ Parent-Child ]                   [ BM25 Lexical ]
   Chunks      Window ]       Chunks                              Chunks
       │           │              │                                   │
       └───────────┴──────────────┴───────────────────────────────────┘
                                       │
                                       ▼
                       [ Canonical RRF Fusion (k=60) ]
                                       │
                                       ▼
                       [ MMR Embedding Diversification ]
                                       │
                                       ▼
                  [ Cross-Encoder Multi-Query Reranker ]
                     (ms-marco-MiniLM-L-6-v2, BS=32)
                                       │
                                       ▼
                         [ Relevance Score Gate ]
                     Score >= -10.5 ?
                      ├── No  ──► [ Clean "Out of Scope" Rejection ]
                      └── Yes ──► [ Top-K Grounded Context Builder ]
                                       │
                                       ▼
                             [ Qwen LLM Generator ]
                              (Non-thinking mode)
                                       │
                                       ▼
                    [ Final Answer + Source Attribution ]
```

---

## Key Features

- **Document-Agnostic & Scalable:** Add any text, PDF, DOCX, or Markdown file to `data/raw/` and rebuild indexes with one command.
- **State-of-the-Art Retrieval Performance:**
  - **Hit@1:** **97.56%**
  - **Hit@3:** **100.0%**
  - **Hit@5:** **100.0%**
  - **MRR:** **0.9858**
- **Sub-3s Latency:** Optimized batch reranking and embedding caching yield median end-to-end latency of ~2.33s.
- **Anti-Hallucination Rejection Gate:** Automatically detects out-of-scope or unanswerable queries before calling the generative LLM.
- **Modern Streamlit Research Dashboard:** Modern ChatGPT-inspired UI featuring step-by-step thinking traces, collapsible citations, score breakdowns, and responsive mobile/desktop layout.
- **Comprehensive Test Suite:** 60 automated unit and integration tests (`pytest`) covering architecture, ingestion, fusion, reranking, and generation.

---

## Quick Start: Clone & Run

### 1. Clone Repository & Checkout Branch

Clone the repository and ensure you are on the `feature/05_hybrid_fusion_and_reranking` branch:

```bash
# Clone the repository
git clone https://github.com/Dhinesh-Manikandan/Hybrid-RAG-Project.git

# Navigate into the project folder
cd Hybrid-RAG-Project

# Switch to the feature branch (if not checked out by default)
git checkout feature/05_hybrid_fusion_and_reranking
```

Verify your current branch:
```bash
git branch --show-current
# Expected output: feature/05_hybrid_fusion_and_reranking
```

---

### 2. Environment Setup

Python **3.10+** (Python 3.11 or 3.12 recommended) is required.

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install all core dependencies, vector stores, deep learning libraries, and the web interface:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (GPU Support):**  
> By default, `requirements.txt` installs PyTorch. If you have an NVIDIA GPU and want CUDA acceleration for faster reranking and LLM generation, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 4. Ingest Documents & Build Indexes

Before querying, index the provided welfare schemes into FAISS and BM25:

```bash
python ingest.py
```

This single command automatically:
1. Discovers all documents in `data/raw/` (14 schemes provided out-of-the-box).
2. Cleans text, extracts sections, and generates structural metadata.
3. Generates 3 chunk representations: **Naive**, **Sentence-Window**, and **Parent-Child**.
4. Encodes passages using `all-MiniLM-L6-v2` and persists FAISS indexes into `data/vector_store/`.
5. Builds and saves the BM25 lexical token corpus.

---

### 5. Launch the Streamlit Web Application

Run the interactive web dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501` to start querying with real-time retrieval traces, candidate inspection, and source-attributed answers.

---

## Usage Modes

### Streamlit Research Web UI

The dashboard provides an intuitive, ChatGPT-styled interface:
- **Interactive Chat:** Ask natural language questions about government schemes.
- **Thinking Process Trace:** Real-time step tracker showing query reformulation, dense/sparse search counts, RRF fusion, Cross-Encoder reranking, and generation status.
- **Transparent Citations:** Expandable citation cards with document title, section heading, source file, and chunk ID.
- **Candidate Inspection:** Full retrieval telemetry showing raw FAISS scores, BM25 ranks, RRF scores, and Cross-Encoder logit scores.

Run with:
```bash
streamlit run app/streamlit_app.py
```

---

### Python API / Programmatic Usage

Integrate `HybridRAG` directly into your Python applications:

```python
from retrieve import HybridRAG

# 1. Initialize the Hybrid-RAG pipeline (loads FAISS, BM25, and models)
rag = HybridRAG()

# 2. Perform retrieval with grounded answer generation
query = "What is the maximum subsidy amount under PMAY-U Credit Linked Subsidy Scheme?"
result = rag.answer(query)

print("\n--- GROUNDED ANSWER ---")
print(result["answer"])

print("\n--- CITED SOURCES ---")
for src in result.get("sources", []):
    print(f"[{src['source_index']}] {src['document_title']} | Section: {src['section']}")
```

To run retrieval only (without LLM generation):
```python
# Returns top-k reranked candidates and score breakdowns
retrieval_output = rag.query("What are the eligibility criteria for PM-KISAN?", generate_answer=False)
rag.display(retrieval_output)
```

---

### CLI Query Mode

You can also run ad-hoc queries directly from the command line:

```bash
python -c "from retrieve import HybridRAG; rag = HybridRAG(); res = rag.answer('Who is eligible for Stand-Up India loans?'); rag.display(res)"
```

---

## Adding Custom Documents (Zero-Code Ingestion)

The pipeline is completely document-agnostic. To add your own documents:

1. **Copy your files** into `data/raw/`:
   ```bash
   cp /path/to/my_document.pdf data/raw/
   # Supported file formats: .txt, .pdf, .docx, .md
   ```

2. **Re-run ingestion**:
   ```bash
   python ingest.py
   ```
   Or force a complete clean rebuild:
   ```bash
   python ingest.py --rebuild
   ```

3. **Query immediately**:
   The retrieval indexes and BM25 corpus are updated automatically. No changes to configuration or retrieval code are needed.

---

## Evaluation & Empirical Benchmarks

### Retrieval Accuracy Metrics

Evaluated on the standardized 82 multi-document test suite (`evaluation/multi_document_queries.json`) comparing baseline and optimized fusion settings:

| Configuration | Fusion Top-K ($K$) | Query Variants ($V$) | Hit@1 | Hit@3 | Hit@5 | MRR | Mean Latency | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Configuration A (Baseline)** | 15 | 3 | 97.56% | 100.0% | 100.0% | 0.9858 | 3.24s | 3.07s | 3.71s |
| **Configuration B (Optimized)** | 10 | 3 | **97.56%** | **100.0%** | **100.0%** | **0.9858** | **2.47s** | **2.33s** | **2.92s** |

*Key Takeaway: Reducing fusion candidates to $K=10$ provides a **~24% reduction in latency** with **zero degradation in retrieval accuracy** (perfect 100% Hit@3 and 100% Hit@5).*

---

### Running Evaluation Scripts

Run the standardized benchmark suite:
```bash
python -m scripts.run_standardized_benchmark
```

Run multi-document corpus evaluation:
```bash
python scripts/evaluate_multidoc.py
```

Run generation response quality evaluation (Faithfulness, Correctness, Relevancy):
```bash
python scripts/evaluate_generation.py
```

Run latency and profiling experiments:
```bash
python scripts/run_latency_experiments.py
python scripts/profile_pipeline.py
```

---

### Running Automated Tests

Run the complete test suite:
```bash
pytest
```
*Executes all 60 tests across chunking, embeddings, indexing, RRF fusion, multi-query Cross-Encoder reranking, relevance gating, and grounded answer generation.*

To run tests with detailed verbosity:
```bash
pytest -v
```

---

## Configuration Reference (`src/config.py`)

All system parameters are centralized in [`src/config.py`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/src/config.py). Key parameters include:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `"all-MiniLM-L6-v2"` | SentenceTransformer embedding model name |
| `EMBEDDING_DIM` | `384` | Dimensionality of dense vectors |
| `RERANKER_MODEL_NAME` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-Encoder model for neural passage reranking |
| `LLM_MODEL_NAME` | `"Qwen/Qwen3-0.6B"` | Default generative LLM for answer synthesis |
| `TARGET_CHUNK_SIZE` | `800` | Target chunk size (characters) for naive chunking |
| `SENTENCE_WINDOW_SIZE` | `2` | Number of surrounding sentences attached as window context |
| `PARENT_CHUNK_SIZE` | `512` | Parent chunk size for hierarchical chunking |
| `CHILD_CHUNK_SIZE` | `128` | Child chunk size indexed into FAISS |
| `RETRIEVAL_TOP_K` | `10` | Initial retrieval candidates per channel |
| `FUSION_TOP_K` | `15` | Candidates pooled after Canonical RRF fusion |
| `FINAL_TOP_K` | `5` | Top reranked passages returned to user / context |
| `RRF_K` | `60` | Constant smoothing denominator for Reciprocal Rank Fusion |
| `MMR_LAMBDA` | `0.7` | Trade-off between relevance and diversity in MMR |
| `ENABLE_QUERY_REFORMULATION` | `True` | Whether multi-query expansion is enabled |
| `NUM_REFORMULATIONS` | `3` | Total perspectives evaluated ($V=3$: original + 2 variants) |
| `RERANK_RELEVANCE_THRESHOLD` | `-10.5` | Threshold logit to gate out unrelated / out-of-scope queries |

---

## Repository Structure

```text
Hybrid-RAG-Project/
│
├── app/
│   └── streamlit_app.py        # Streamlit web application & research dashboard
│
├── data/
│   ├── raw/                    # Raw text documents (14 Indian government schemes)
│   ├── processed/              # Cleaned text and parsed sections
│   ├── vector_store/           # FAISS indexes (.index) and metadata (.pkl)
│   └── retrieval/              # Cached retrieval outputs & run histories
│
├── evaluation/
│   ├── multi_document_queries.json  # 82 standardized benchmark test queries
│   ├── generation_queries.json      # Grounded generation evaluation queries
│   ├── ground_truth.json            # Reference ground-truth scheme facts
│   ├── results/                     # Empirical benchmark JSON and CSV results
│   └── response_quality/            # Faithfulness, Relevancy, and Correctness metrics
│
├── notebooks/                  # Interactive experimentation & development
│   ├── 01_setup.ipynb
│   ├── 02_document_processing.ipynb
│   ├── 03_embeddings_and_vector_store.ipynb
│   ├── 04_retrieval.ipynb
│   └── 05_hybrid_fusion_and_reranking.ipynb
│
├── scripts/                    # Benchmarking, profiling, and ablation scripts
│   ├── run_standardized_benchmark.py
│   ├── evaluate_multidoc.py
│   ├── evaluate_generation.py
│   ├── run_latency_experiments.py
│   ├── profile_pipeline.py
│   └── run_reformulation_ablation.py
│
├── src/                        # Modular source code
│   ├── chunking/               # Naive, Sentence-Window, and Parent-Child chunkers
│   ├── embeddings/             # SentenceTransformer encoding helpers
│   ├── evaluation/             # Metrics calculators (MRR, Hit@K, NDCG)
│   ├── fusion/                 # Canonical RRF and MMR diversification
│   ├── generation/             # Context builder & Qwen LLM answer generator
│   ├── indexing/               # FAISS vector store builders & loaders
│   ├── ingestion/              # Document loaders, cleaners, title extractors
│   ├── reranking/              # Multi-query CrossEncoder reranker
│   ├── retrieval/              # Dense retriever, BM25, query reformulation
│   ├── utils/                  # I/O utilities
│   └── config.py               # Central configuration settings
│
├── tests/                      # 60 automated unit & integration tests
│   ├── test_architecture.py
│   ├── test_crossencoder_multiquery.py
│   ├── test_document_agnostic.py
│   ├── test_generation.py
│   ├── test_query_reformulation.py
│   └── test_simple_evaluator.py
│
├── ingest.py                   # Master ingestion pipeline CLI
├── retrieve.py                 # Master HybridRAG retrieval & answer generation API
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## Preloaded Schemes

The system comes pre-indexed with detailed guidelines for 14 national welfare programs:
1. **Atal Pension Yojana (APY)** — Guaranteed pension scheme for unorganized sector workers.
2. **Pradhan Mantri Awas Yojana - Gramin (PMAY-G)** — Rural housing assistance scheme.
3. **Pradhan Mantri Awas Yojana - Urban (PMAY-U)** — Urban affordable housing and interest subsidies.
4. **PM-KISAN** — Income support for farmer families ($₹6,000$ per year).
5. **PM SVANidhi** — Micro-credit working capital loans for street vendors.
6. **PM Vishwakarma** — End-to-end holistic support for traditional artisans and craftspeople.
7. **Pradhan Mantri Ujjwala Yojana (PMUY)** — LPG connection subsidies for rural women.
8. **Pradhan Mantri Jan Dhan Yojana (PMJDY)** — Universal banking access, overdraft, and RuPay card insurance.
9. **PM Kaushal Vikas Yojana (PMKVY)** — Skill certification and vocational training for youth.
10. **Pradhan Mantri Mudra Yojana (PMMY)** — Non-collateral business loans under Shishu, Kishore, and Tarun.
11. **Pradhan Mantri Fasal Bima Yojana (PMFBY)** — Comprehensive crop insurance for farmers.
12. **Stand-Up India Scheme** — Bank loans ($₹10\text{ lakh}$ to $₹1\text{ crore}$) for SC/ST and women entrepreneurs.
13. **Sukanya Samriddhi Yojana (SSY)** — High-interest savings scheme for girl child education and welfare.
14. **Pradhan Mantri Garib Kalyan Anna Yojana (PMGKAY)** — Free food grain entitlement program.

---

<br>

## 📖 License & Citation

<p align="left">
  <img src="https://img.shields.io/badge/Purpose-Academic%20%26%20Applied%20Research-2563eb?style=for-the-badge" alt="Purpose">
  <img src="https://img.shields.io/badge/Year-2026-16a34a?style=for-the-badge" alt="Year">
  <img src="https://img.shields.io/badge/Topic-Advanced%20RAG-9333ea?style=for-the-badge" alt="Topic">
</p>

> [!NOTE]
> This project is developed for **academic and applied research** in advanced RAG architectures.
> If you use it in your research, please cite it using one of the formats below.

### 📌 Cite this work

| | |
|---|---|
| **Author** | Dhinesh Manikandan |
| **Title** | *Hybrid-RAG: Multi-Representation Retrieval, Reciprocal Rank Fusion, and Grounded Generation* |
| **Year** | 2026 |
| **Publisher** | GitHub |
| **URL** | [github.com/Dhinesh-Manikandan/Hybrid-RAG-Project](https://github.com/Dhinesh-Manikandan/Hybrid-RAG-Project) |

<details open>
<summary><b>📄 BibTeX</b></summary>

```bibtex
@misc{manikandan2026hybridrag,
  author       = {Dhinesh Manikandan},
  title        = {Hybrid-RAG: Multi-Representation Retrieval, Reciprocal Rank Fusion, and Grounded Generation},
  year         = {2026},
  howpublished = {\url{https://github.com/Dhinesh-Manikandan/Hybrid-RAG-Project}}
}
```

</details>

<details>
<summary><b>📝 APA</b></summary>

```text
Manikandan, D. (2026). Hybrid-RAG: Multi-Representation Retrieval, Reciprocal Rank Fusion, and Grounded Generation. GitHub. https://github.com/Dhinesh-Manikandan/Hybrid-RAG-Project
```

</details>

<details>
<summary><b>🏛️ IEEE</b></summary>

```text
D. Manikandan, "Hybrid-RAG: Multi-Representation Retrieval, Reciprocal Rank Fusion, and Grounded Generation," GitHub, 2026. [Online]. Available: https://github.com/Dhinesh-Manikandan/Hybrid-RAG-Project
```

</details>

<br>

<p align="center">
  <sub>⭐ If this project helped your research, consider giving it a star!</sub>
</p>
