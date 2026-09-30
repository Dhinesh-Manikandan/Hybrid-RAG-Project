"""
cross_encoder.py -- CrossEncoder reranking with multi-query evidence aggregation.

The reranker receives (query, candidate_text) pairs only.
It has zero knowledge of the domain, document, or section.
All candidate metadata is preserved and extended with rerank_score.

Supports:
1. Multi-query mean aggregation:
   rerank_score(c) = mean([CE(q_1, c), CE(q_2, c), ...])
2. Single-query execution when query_variants is None or has length 1.
3. Safe fallback to single-query reranking if multi-query batch scoring encounters an error.
"""

import logging
from typing import Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker_cache: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str) -> CrossEncoder:
    """Load and cache a CrossEncoder model."""
    if model_name not in _reranker_cache:
        print(f"Loading reranker: {model_name}")
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def rerank(
    query: str,
    candidates: list[dict],
    model_name: str,
    top_k: int = 5,
    query_variants: Optional[list[str]] = None,
) -> list[dict]:
    """
    Rerank candidates using a CrossEncoder with multi-query mean aggregation.

    The reranker only sees (query, text) pairs — no domain information.

    Args:
        query:          Original user query string.
        candidates:     List of result dicts, each must have a "text" or "retrieval_text" field.
        model_name:     CrossEncoder model name.
        top_k:          Number of top results to return.
        query_variants: Optional list of query reformulations (including original query).
                        If provided and len > 1, candidates are scored against all variants
                        and aggregated via arithmetic mean. Defaults to None (single-query).

    Returns:
        List of up to top_k result dicts, each with "rerank_score" added.
        Sorted by rerank_score descending.
    """
    if not candidates:
        return []

    reranker = get_reranker(model_name)
    candidate_texts = [c.get("retrieval_text", c["text"]) for c in candidates]
    num_candidates = len(candidates)

    # Normalize query variants list
    variants: list[str] = []
    if query_variants:
        # Filter empty strings and deduplicate while preserving order
        seen = set()
        for v in query_variants:
            v_clean = v.strip() if isinstance(v, str) else ""
            if v_clean and v_clean not in seen:
                seen.add(v_clean)
                variants.append(v_clean)

    # Ensure original query is present as the primary anchor
    if not variants:
        variants = [query]
    elif query not in variants:
        variants = [query] + variants

    # Multi-query vs Single-query execution
    if len(variants) > 1:
        try:
            # Construct all V x N pairs in a single batch
            # Outer loop: variants, Inner loop: candidates
            all_pairs = [(v, text) for v in variants for text in candidate_texts]
            raw_scores = reranker.predict(all_pairs)

            # Compute arithmetic mean per candidate across all variants
            num_variants = len(variants)
            reranked = []
            for i, candidate in enumerate(candidates):
                candidate_variant_scores = [
                    float(raw_scores[v * num_candidates + i])
                    for v in range(num_variants)
                ]
                mean_score = sum(candidate_variant_scores) / num_variants

                result = candidate.copy()
                result["rerank_score"] = float(mean_score)
                result["rerank_scores_by_variant"] = candidate_variant_scores
                reranked.append(result)

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        except Exception as exc:
            logger.warning(
                f"Multi-query CrossEncoder reranking failed ({exc}); falling back to single-query reranking."
            )
            # Fall back safely to single-query execution below

    # Single-query baseline / fallback
    pairs = [(query, text) for text in candidate_texts]
    scores = reranker.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["rerank_score"] = float(score)
        reranked.append(result)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
