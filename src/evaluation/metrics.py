"""
metrics.py -- Document-agnostic evaluation utilities.

Evaluation data lives in evaluation/ground_truth.json.
Format:
[
  {
    "query": "What are the eligibility criteria?",
    "relevant_document_ids": ["doc_4a8f2c91e3b1"],
    "relevant_chunk_ids": ["doc_4a8f2c91e3b1__naive__2", ...]
  },
  ...
]

No hardcoded queries, sections, or document names here.
Adding a new document means adding entries to ground_truth.json only.
"""

import json
from pathlib import Path


def load_ground_truth(path: Path) -> list[dict]:
    """
    Load evaluation ground truth from a JSON file.

    Args:
        path: Path to ground_truth.json.

    Returns:
        List of ground-truth dicts with keys:
        "query", "relevant_document_ids", "relevant_chunk_ids"
    """
    if not path.exists():
        print(f"Warning: ground truth file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} evaluation queries from {path}")
    return data


def hit_at_k(
    results: list[dict],
    relevant_document_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> bool:
    """
    Check if any of the top-k results match a relevant document or chunk.

    Matching logic (generic, document-agnostic):
    - A result is a hit if its document_id is in relevant_document_ids
    - OR if its chunk_id or source_chunk_id is in relevant_chunk_ids

    Args:
        results:               Ranked list of result dicts.
        relevant_document_ids: Set of relevant document IDs.
        relevant_chunk_ids:    Set of relevant chunk IDs.
        k:                     Number of top results to check.

    Returns:
        True if any of the top-k results is relevant.
    """
    top_k = results[:k]

    rel_docs   = set(relevant_document_ids)
    rel_chunks = set(relevant_chunk_ids)

    for result in top_k:
        if result.get("document_id") in rel_docs:
            return True
        if result.get("chunk_id") in rel_chunks:
            return True
        if result.get("source_chunk_id") in rel_chunks:
            return True

    return False


def evaluate(
    retrieval_fn,
    ground_truth: list[dict],
    ks: list[int] = None,
) -> dict:
    """
    Evaluate a retrieval function against ground truth.

    Args:
        retrieval_fn:  Callable(query: str) -> list[dict].
                       Must return result dicts with "document_id" and "chunk_id".
        ground_truth:  List of GT dicts from load_ground_truth().
        ks:            List of k values for Hit@K. Default [1, 3, 5].

    Returns:
        Dict with per-query results and aggregate Hit@K metrics.
    """
    if ks is None:
        ks = [1, 3, 5]

    rows = []

    for entry in ground_truth:
        query        = entry["query"]
        rel_doc_ids  = entry.get("relevant_document_ids", [])
        rel_chunk_ids = entry.get("relevant_chunk_ids", [])

        results = retrieval_fn(query)

        row = {"query": query}
        for k in ks:
            row[f"hit_at_{k}"] = hit_at_k(
                results, rel_doc_ids, rel_chunk_ids, k
            )

        rows.append(row)

    # Aggregate
    summary = {}
    for k in ks:
        key = f"hit_at_{k}"
        hits = [r[key] for r in rows]
        summary[f"Hit@{k}"] = sum(hits) / len(hits) if hits else 0.0

    return {
        "per_query": rows,
        "summary":   summary,
    }
