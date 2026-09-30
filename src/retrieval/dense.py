"""
dense.py -- Generic dense retrieval from FAISS indexes.

No document-specific logic. Works across all documents simultaneously
because all documents share the same indexes.
"""

import numpy as np
import faiss

from src.embeddings.encoder import encode_query


def retrieve_dense(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    model_name: str,
    k: int = 10,
) -> list[dict]:
    """
    Retrieve top-k candidates from a FAISS index for a query.

    The index contains vectors from ALL documents. The query
    automatically searches across all of them.

    Args:
        query:      Query string.
        index:      FAISS index (naive, sentence_window, or parent_child).
        metadata:   List of metadata dicts aligned with index positions.
        model_name: Embedding model name (same as used during indexing).
        k:          Number of candidates to retrieve.

    Returns:
        List of result dicts, each with:
        {
            "rank":           int,
            "retrieval_score": float,
            "retrieval_type":  str,
            ...all fields from the metadata dict...
        }
    """
    if index.ntotal == 0:
        return []

    # Encode query
    query_embedding = encode_query(query, model_name)

    # Search FAISS
    actual_k = min(k, index.ntotal)
    scores, indices = index.search(query_embedding, actual_k)

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if idx == -1:
            continue

        meta = metadata[idx]

        result = {
            "rank":            rank,
            "retrieval_score": float(score),
        }
        result.update(meta)  # Spread all metadata fields into result

        results.append(result)

    return results
