# Hybrid RAG Project

A Hybrid Retrieval-Augmented Generation (RAG) system combining dense vector search and sparse keyword search (BM25) with reranking to deliver highly accurate context retrieval for AI models.

## Project Structure

```text
hybrid-rag-project/
│
├── notebooks/
│   └── 01_setup.ipynb     # Initial setup and experimentation notebooks
│
├── src/                    # Core source code (retrievers, embedders, pipelines)
├── data/                   # Raw and processed datasets
├── evaluation/             # Evaluation scripts and metrics (RAGAS, hit rate, MRR)
├── app/                    # Web interface / API service (Streamlit / FastAPI)
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
└── .gitignore              # Ignored files and folders
```

## Getting Started

1. **Clone & Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Initial Notebook**
   Open [`notebooks/01_setup.ipynb`](file:///d:/Sem%20VII/AI/Project/Implementation/Hybrid-RAG-Project/notebooks/01_setup.ipynb) in Jupyter or VS Code to verify the setup.