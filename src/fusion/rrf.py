"""
rrf.py -- Reciprocal Rank Fusion and MMR diversification.

Key design:
1. canonical_rrf():
   - Groups results by (document_id, source_chunk_id) before computing RRF scores.
   - This prevents overlapping sentence-window/parent-child representations
     from generating multiple independent RRF votes for the same content.
   - Genuinely different content from different naive chunks receives its own vote.
   - Different retrieval strategies finding the same chunk REINFORCE each other.

2. mmr_diversify():
   - Embedding-based Maximal Marginal Relevance.
   - No section names, no domain knowledge, no document-specific parameters.
   - Works on any document from any domain.
"""

from collections import defaultdict
from typing import Optional

import numpy as np


# ============================================================
# CANONICAL RRF
# ============================================================

def canonical_rrf(
    retrieval_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion with canonical grouping.

    Canonical identity = (document_id, source_chunk_id).
    This key is used instead of raw text so that:
    - Multiple overlapping SW/PC representations of the same naive chunk
      are treated as ONE logical unit per retrieval list.
    - Different naive chunks (different content) remain distinct.
    - Contributions from multiple retrieval strategies are summed.

    RRF formula: score(c) = sum over lists L: 1 / (k + rank_in_L(c))

    Args:
        retrieval_lists: List of retrieval result lists.
                         Each sublist is from one retrieval strategy.
                         Each item must have "document_id" and "source_chunk_id".
        k:               RRF smoothing constant (default 60).

    Returns:
        Sorted list of fused result dicts with "rrf_score" field added.
    """
    # rrf_scores maps canonical_id -> cumulative RRF score
    rrf_scores: dict[str, float] = defaultdict(float)

    # best_representative maps canonical_id -> best result dict seen so far
    best_representative: dict[str, dict] = {}

    for result_list in retrieval_lists:
        # Within each list, deduplicate by canonical_id and take best rank
        canonical_rank = _deduplicate_within_list(result_list)

        for canonical_id, (rank, result) in canonical_rank.items():
            rrf_scores[canonical_id] += 1.0 / (k + rank)

            # Track the highest-quality representative for this canonical group
            # Prefer the representative with the best retrieval_score
            existing = best_representative.get(canonical_id)
            if existing is None or result.get("retrieval_score", 0) > existing.get("retrieval_score", 0):
                best_representative[canonical_id] = result

    # Build fused result list
    from src.chunking.representation import build_retrieval_text

    fused = []
    for canonical_id, rrf_score in rrf_scores.items():
        rep = best_representative[canonical_id].copy()
        rep["rrf_score"]    = rrf_score
        rep["canonical_id"] = canonical_id
        # Parent-child represents a child of a larger unit; restore parent text for context
        if rep.get("retrieval_type") == "parent_child" and "parent_text" in rep.get("extra", {}):
            rep["child_text"] = rep["text"]
            rep["text"] = rep["extra"]["parent_text"]
            rep["retrieval_text"] = build_retrieval_text(rep, rep["text"])
        # Collect all contributing retrieval types for traceability
        rep.setdefault("retrieval_sources", [])
        fused.append(rep)

    # Annotate retrieval_sources and query_variants
    for result_list in retrieval_lists:
        for r in result_list:
            cid = _get_canonical_id(r)
            rtype = r.get("retrieval_type", "unknown")
            qvar = r.get("query_variant")
            for item in fused:
                if item["canonical_id"] == cid:
                    if rtype not in item["retrieval_sources"]:
                        item["retrieval_sources"].append(rtype)
                    if qvar:
                        item.setdefault("query_variants", [])
                        if qvar not in item["query_variants"]:
                            item["query_variants"].append(qvar)

    # Sort by RRF score descending
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused


def _get_canonical_id(result: dict) -> str:
    """
    Compute the canonical identity for a retrieval result.

    canonical_id = "{document_id}::{source_chunk_id}"

    Both fields are guaranteed by the universal metadata schema.
    """
    doc_id    = result.get("document_id", "unknown_doc")
    chunk_id  = result.get("source_chunk_id", result.get("chunk_id", "unknown_chunk"))
    return f"{doc_id}::{chunk_id}"


def _deduplicate_within_list(
    result_list: list[dict],
) -> dict[str, tuple[int, dict]]:
    """
    Within a single retrieval list, deduplicate by canonical_id.

    For duplicates, keep the one with the best (lowest) rank.

    Returns:
        Dict mapping canonical_id -> (best_rank, best_result_dict)
    """
    canonical_rank: dict[str, tuple[int, dict]] = {}

    for rank, result in enumerate(result_list, start=1):
        cid = _get_canonical_id(result)

        if cid not in canonical_rank or rank < canonical_rank[cid][0]:
            canonical_rank[cid] = (rank, result)

    return canonical_rank


# ============================================================
# MMR DIVERSIFICATION
# ============================================================

def mmr_diversify(
    candidates: list[dict],
    query_embedding: np.ndarray,
    model_name: str,
    top_k: int = 15,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance (MMR) diversification.

    MMR score = lambda * relevance(c) - (1 - lambda) * max_sim(c, selected)

    Where:
    - relevance(c)        = cosine similarity between query and candidate embedding
    - max_sim(c, selected) = max cosine similarity between c and already-selected candidates
    - lambda_param        = trade-off: 1.0 = pure relevance, 0.0 = pure diversity

    This is purely embedding-based and domain-agnostic.
    No section names, no document-specific config.

    Args:
        candidates:      List of fused result dicts (must have "text" field).
        query_embedding: Query embedding, shape (1, dim).
        model_name:      Embedding model name (for encoding candidates).
        top_k:           Number of results to return.
        lambda_param:    Relevance vs diversity trade-off.

    Returns:
        Reordered list of up to top_k candidates.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    from src.embeddings.encoder import encode_texts

    # Encode all candidates
    texts = [c["text"] for c in candidates]
    candidate_embeddings = encode_texts(texts, model_name, show_progress=False)

    # Normalize query embedding
    q = query_embedding[0].copy()
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # Compute relevance scores (cosine similarity to query)
    relevance = candidate_embeddings @ q  # shape (N,)

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(top_k):
        if not remaining_indices:
            break

        if not selected_indices:
            # First selection: pick most relevant candidate
            best_idx = max(remaining_indices, key=lambda i: relevance[i])
        else:
            # MMR: balance relevance and diversity
            selected_embeddings = candidate_embeddings[selected_indices]  # (S, dim)

            mmr_scores = []
            for i in remaining_indices:
                emb_i = candidate_embeddings[i]  # (dim,)

                # Similarity to already-selected candidates
                sim_to_selected = selected_embeddings @ emb_i  # (S,)
                max_sim = float(np.max(sim_to_selected))

                mmr = lambda_param * float(relevance[i]) - (1 - lambda_param) * max_sim
                mmr_scores.append((i, mmr))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]
