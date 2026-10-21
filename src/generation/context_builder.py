"""
context_builder.py -- Generic, document-agnostic LLM context builder for Hybrid-RAG.

Transforms final ranked retrieval candidates into structured, LLM-ready context
while preserving source provenance and metadata for attribution.

Key feature: query-aware scope filtering prunes chunks from schemes not referenced
by the user's query before they ever reach the LLM, preventing cross-scheme contamination.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import GENERATION_CONTEXT_TOP_K


def format_source_document_title(candidate: Dict[str, Any]) -> str:
    """
    Extract or derive a readable document title from candidate metadata.
    Domain-agnostic fallback to source_file or document_id.
    """
    doc_title = candidate.get("document_title")
    if doc_title:
        return str(doc_title).strip()

    source_file = candidate.get("source_file")
    if source_file:
        return Path(source_file).stem.replace("_", " ").title()

    doc_id = candidate.get("document_id")
    if doc_id:
        return str(doc_id).strip()

    return "Unknown Document"


def format_source_section(candidate: Dict[str, Any]) -> str:
    """
    Format section hierarchy if heading_path is present, otherwise fallback to section.
    """
    heading_path = candidate.get("heading_path")
    if heading_path and isinstance(heading_path, list):
        valid_headings = [str(h).strip() for h in heading_path if str(h).strip()]
        if valid_headings:
            return " > ".join(valid_headings)

    section = candidate.get("section")
    if section and str(section).strip() and str(section).strip() != "DOCUMENT":
        return str(section).strip()

    return "GENERAL"


def format_source_entry(index: int, candidate: Dict[str, Any]) -> str:
    """
    Format an individual retrieval candidate into a structured source block.

    Example:
        [Source 1]
        Document: PMAY-U
        Section: ELIGIBILITY
        Source File: pmay_u.txt

        Content:
        ...
    """
    doc_title = format_source_document_title(candidate)
    section_str = format_source_section(candidate)
    source_file = candidate.get("source_file", "N/A")
    content = candidate.get("text", "").strip()

    return (
        f"[Source {index}]\n"
        f"Document: {doc_title}\n"
        f"Section: {section_str}\n"
        f"Source File: {source_file}\n\n"
        f"Content:\n"
        f"{content}"
    )


def extract_source_metadata(index: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured source attribution dictionary from a retrieval candidate.
    Preserves document_id, source_file, section, heading_path, chunk_id.
    """
    return {
        "source_index": index,
        "document_id": candidate.get("document_id", "N/A"),
        "document_title": format_source_document_title(candidate),
        "source_file": candidate.get("source_file", "N/A"),
        "section": candidate.get("section", "N/A"),
        "heading_path": candidate.get("heading_path", []),
        "chunk_id": candidate.get("chunk_id", candidate.get("source_chunk_id", "N/A")),
        "retrieval_type": candidate.get("retrieval_type", "N/A"),
        "rerank_score": candidate.get("rerank_score"),
    }


# ---------------------------------------------------------------------------
# Scheme vocabulary: maps lowercase query tokens → canonical document title tokens
# Add more aliases here when new scheme documents are ingested.
# ---------------------------------------------------------------------------
_SCHEME_ALIASES: Dict[str, List[str]] = {
    "pmay": ["pradhan mantri awas", "pmay"],
    "pmay-u": ["pradhan mantri awas yojana - urban", "pmay-u", "pmay urban"],
    "pmay-g": ["pradhan mantri awas yojana - gramin", "pmay-g", "pmay gramin", "pmay rural"],
    "pm-kisan": ["pradhan mantri kisan samman nidhi", "pm-kisan", "pm kisan"],
    "kisan": ["pradhan mantri kisan samman nidhi", "pm-kisan", "pm kisan"],
    "apy": ["atal pension yojana", "apy"],
    "atal pension": ["atal pension yojana", "apy"],
    "svani": ["pm svanidhi", "pm street vendor", "svanidhi", "svani"],
    "svanidhi": ["pm svanidhi", "pm street vendor", "svanidhi"],
    "stand-up india": ["stand-up india", "stand up india"],
    "standup": ["stand-up india", "stand up india"],
    "pm vishwakarma": ["pm vishwakarma", "vishwakarma"],
    "vishwakarma": ["pm vishwakarma", "vishwakarma"],
    "pm-jay": ["ayushman bharat", "pm-jay", "pmjay"],
    "ayushman": ["ayushman bharat", "pm-jay", "pmjay"],
    "pmjdy": ["pradhan mantri jan dhan", "pmjdy", "jan dhan"],
    "jan dhan": ["pradhan mantri jan dhan", "pmjdy", "jan dhan"],
}


def _extract_query_scheme_tokens(query: str) -> List[str]:
    """
    Extract the canonical title tokens for any scheme(s) mentioned in the query.
    Returns an empty list if no scheme is detected (broad / multi-scheme query).
    """
    q_lower = query.lower()
    matched_tokens: List[str] = []
    for alias, canonical_tokens in _SCHEME_ALIASES.items():
        if alias in q_lower:
            matched_tokens.extend(canonical_tokens)
    return list(set(matched_tokens))


def _candidate_matches_scope(candidate: Dict[str, Any], scope_tokens: List[str]) -> bool:
    """
    Return True if this candidate belongs to a scheme that is within query scope.
    Checks document_title and document_id (lowercased) against scope_tokens.
    """
    doc_title = (candidate.get("document_title") or "").lower()
    doc_id = (candidate.get("document_id") or "").lower()
    haystack = doc_title + " " + doc_id
    return any(tok in haystack for tok in scope_tokens)


def filter_candidates_by_query_scope(
    candidates: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """
    Prune candidates that belong to schemes NOT referenced by the query.

    Strategy:
    1. Detect which scheme(s) the query mentions.
    2. If at least one scheme is identified, keep only candidates whose
       document_title/document_id matches those schemes.
    3. If no scheme is detected (broad question) or the filter removes everything,
       fall back to the original full candidate list.

    This prevents small LLMs from mixing unrelated scheme content even when
    an unrelated chunk happens to score well in retrieval.
    """
    if not candidates or not query:
        return candidates

    scope_tokens = _extract_query_scheme_tokens(query)
    if not scope_tokens:
        # No specific scheme detected — broad question, keep everything
        return candidates

    filtered = [c for c in candidates if _candidate_matches_scope(c, scope_tokens)]
    if not filtered:
        # Scope filter removed everything (document naming mismatch?) — fall back
        return candidates

    return filtered


def build_context(
    candidates: Optional[List[Dict[str, Any]]],
    top_k: int = GENERATION_CONTEXT_TOP_K,
    score_margin_threshold: float = 5.0,
    query: Optional[str] = None,
    filter_unrelated: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build structured LLM context and source attribution records from ranked candidates.

    Two-stage filtering (applied ONLY when filter_unrelated is True, typically for Hybrid RAG):
    1. Query-scope filter: drops chunks from unrelated schemes when a specific scheme
       is detectable in the query (prevents cross-scheme contamination in small LLMs).
    2. Score-margin filter: drops low-scoring outlier noise passages with a large
       rerank_score drop relative to the top-ranked candidate.

    When filter_unrelated is False (e.g. for baseline RAG strategies like Naive RAG,
    Sentence Window, Parent-Child), chunks are passed 'as is' without pruning unrelated
    schemes or outlier scores.

    Args:
        candidates: List of candidate dicts.
        top_k: Maximum number of context sources to include (default: GENERATION_CONTEXT_TOP_K).
        score_margin_threshold: Maximum allowed drop in rerank_score from the top candidate.
                                 Candidates scoring worse than (max_score - threshold) are filtered.
        query: Optional user query string. When provided and filter_unrelated is True,
               enables query-scope filtering.
        filter_unrelated: Whether to filter unrelated chunks (True for Hybrid RAG, False
                          for baseline RAGs to preserve raw/unfiltered chunks).

    Returns:
        Tuple of (context_str, sources_metadata_list)
        If candidates is empty or None, returns ("", []).
    """
    if not candidates:
        return "", []

    if filter_unrelated:
        # Stage 1: query-aware scheme scope filtering
        if query:
            candidates = filter_candidates_by_query_scope(candidates, query)

        # Stage 2: Adaptive score-margin filtering
        scores = [c.get("rerank_score") for c in candidates if c.get("rerank_score") is not None]
        if scores and len(scores) > 1:
            top_score = max(scores)
            filtered_candidates = [
                c for c in candidates
                if c.get("rerank_score") is None or (top_score - c.get("rerank_score")) <= score_margin_threshold
            ]
            if not filtered_candidates:
                filtered_candidates = [candidates[0]]
        else:
            filtered_candidates = candidates
    else:
        filtered_candidates = candidates

    selected = filtered_candidates[:top_k]
    source_blocks: List[str] = []
    sources_meta: List[Dict[str, Any]] = []

    for idx, cand in enumerate(selected, start=1):
        block = format_source_entry(idx, cand)
        source_blocks.append(block)
        sources_meta.append(extract_source_metadata(idx, cand))

    context_str = "\n\n".join(source_blocks)
    return context_str, sources_meta



def build_context_string(
    candidates: Optional[List[Dict[str, Any]]],
    top_k: int = GENERATION_CONTEXT_TOP_K,
    query: Optional[str] = None,
    filter_unrelated: bool = True,
) -> str:
    """Convenience helper returning only the formatted context text."""
    context_str, _ = build_context(
        candidates,
        top_k=top_k,
        query=query,
        filter_unrelated=filter_unrelated,
    )
    return context_str


def extract_sources(
    candidates: Optional[List[Dict[str, Any]]],
    top_k: int = GENERATION_CONTEXT_TOP_K,
    query: Optional[str] = None,
    filter_unrelated: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience helper returning only the structured source metadata."""
    _, sources = build_context(
        candidates,
        top_k=top_k,
        query=query,
        filter_unrelated=filter_unrelated,
    )
    return sources
