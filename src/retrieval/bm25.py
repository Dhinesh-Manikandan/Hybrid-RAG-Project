"""
bm25.py -- BM25 lexical retrieval.

Operates on naive chunks as the canonical BM25 textual unit.
Uses the canonical retrieval_text representation (document + heading context).
All naive chunks from all documents are included automatically.
No document-specific configuration required.
"""

import re
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import BM25_CORPUS_PATH


class BM25Retriever:
    """
    BM25 retriever backed by naive chunks from all documents.

    The corpus is built from naive chunks' retrieval_text fields.
    Tokenization is generic alphanumeric word tokenization with lowercasing.
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.corpus_metadata: list[dict] = []

    def build(self, naive_chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of naive chunk dicts.

        Args:
            naive_chunks: List of chunk dicts with "text" and optional "retrieval_text".
                          All documents' naive chunks should be included.
        """
        self.corpus_metadata = naive_chunks
        tokenized = [self._tokenize(c.get("retrieval_text", c["text"])) for c in naive_chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(naive_chunks)} documents")

    def save(self, path: Path = BM25_CORPUS_PATH) -> None:
        """Save BM25 state and corpus metadata to disk."""
        payload = {
            "bm25":            self.bm25,
            "corpus_metadata": self.corpus_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"BM25 corpus saved: {path}")

    def load(self, path: Path = BM25_CORPUS_PATH) -> None:
        """Load BM25 state and corpus metadata to disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 corpus not found: {path}\n"
                f"Run: python ingest.py"
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.corpus_metadata = payload["corpus_metadata"]
        print(f"BM25 corpus loaded: {len(self.corpus_metadata)} documents")

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        Retrieve top-k results using BM25.

        Args:
            query: Query string.
            k:     Number of results to return.

        Returns:
            List of result dicts with universal metadata + bm25 score.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load().")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        k = min(k, len(self.corpus_metadata))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            meta = self.corpus_metadata[idx]
            result = {
                "rank":            rank,
                "retrieval_score": float(scores[idx]),
                "retrieval_type":  "bm25",
            }
            result.update(meta)
            results.append(result)

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Alphanumeric tokenization with lowercasing and punctuation stripping.
        Generic — works on any document domain.
        """
        return re.findall(r"\b\w+\b", text.lower())
