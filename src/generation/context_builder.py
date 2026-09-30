"""
context_builder.py -- Generic, document-agnostic LLM context builder for Hybrid-RAG.

Transforms final ranked retrieval candidates into structured, LLM-ready context
while preserving source provenance and metadata for attribution.
"""

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


def build_context(
    candidates: Optional[List[Dict[str, Any]]],
    top_k: int = GENERATION_CONTEXT_TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build structured LLM context and source attribution records from ranked candidates.

    Args:
        candidates: List of reranked candidate dicts (e.g. from final_results).
        top_k: Maximum number of context sources to include (default: GENERATION_CONTEXT_TOP_K).

    Returns:
        Tuple of (context_str, sources_metadata_list)
        If candidates is empty or None, returns ("", []).
    """
    if not candidates:
        return "", []

    selected = candidates[:top_k]
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
) -> str:
    """Convenience helper returning only the formatted context text."""
    context_str, _ = build_context(candidates, top_k=top_k)
    return context_str


def extract_sources(
    candidates: Optional[List[Dict[str, Any]]],
    top_k: int = GENERATION_CONTEXT_TOP_K,
) -> List[Dict[str, Any]]:
    """Convenience helper returning only the structured source metadata."""
    _, sources = build_context(candidates, top_k=top_k)
    return sources
