"""
parent_child.py -- Parent-Child chunking strategy.

For each naive chunk (parent), generates smaller child chunks.
Children are embedded for retrieval; the parent text is returned to the user.

Every child links back to its parent and to the naive chunk via source_chunk_id,
enabling canonical RRF grouping.

Completely generic: no hardcoded section names or document-specific logic.
"""
from typing import Optional


def build_parent_child_chunks(
    document_id: str,
    source_file: str,
    naive_chunks: list[dict],
    document_title: Optional[str] = None,
    child_chunk_size: int = 128,
    child_chunk_overlap: int = 16,
) -> list[dict]:
    """
    Build parent-child representations from naive chunks.

    For each naive chunk (parent), split its text into smaller child chunks.
    The child text is used for embedding and retrieval.
    The parent text is stored in metadata for user-facing results.

    Args:
        document_id:        Stable document identifier.
        source_file:        Original filename.
        naive_chunks:       List of naive chunk dicts (the parents).
        document_title:     Canonical document title.
        child_chunk_size:   Target child chunk size in characters.
        child_chunk_overlap: Overlap between child chunks in characters.

    Returns:
        List of child chunk dicts with universal metadata schema.
    """
    from pathlib import Path
    from src.chunking.representation import build_retrieval_text

    if not document_title:
        if naive_chunks and naive_chunks[0].get("document_title"):
            document_title = naive_chunks[0]["document_title"]
        elif source_file:
            document_title = Path(source_file).stem.replace("_", " ").replace("-", " ").title()

    results = []

    for parent in naive_chunks:
        parent_chunk_id = parent["chunk_id"]
        parent_text     = parent["text"]
        section_name    = parent["section"]
        heading_path    = parent.get("heading_path", [section_name])
        doc_title       = parent.get("document_title", document_title)

        # Split parent into children
        children = _split_into_children(
            parent_text,
            child_chunk_size,
            child_chunk_overlap,
        )

        for child_position, child_text in enumerate(children):
            if not child_text.strip():
                continue

            child_id = f"{parent_chunk_id}__child__{child_position}"

            chunk_dict = {
                "document_id":     document_id,
                "source_file":     source_file,
                "document_title":  doc_title,
                "chunk_id":        child_id,
                "retrieval_type":  "parent_child",
                "text":            child_text,
                # All children of this parent share the same source_chunk_id
                # -> they form a canonical group in RRF
                "source_chunk_id": parent_chunk_id,
                "parent_chunk_id": parent_chunk_id,
                "section":         section_name,
                "heading_path":    heading_path,
                "extra": {
                    "parent_text":    parent_text,
                    "child_position": child_position,
                    "n_children":     len(children),
                },
            }
            chunk_dict["retrieval_text"] = build_retrieval_text(chunk_dict, child_text)
            results.append(chunk_dict)

    return results


def _split_into_children(
    text: str,
    child_size: int,
    child_overlap: int,
) -> list[str]:
    """
    Split a parent text into overlapping child chunks.
    """
    children = []
    start = 0

    while start < len(text):
        end = start + child_size
        chunk = text[start:end].strip()
        if chunk:
            children.append(chunk)
        step = max(1, child_size - child_overlap)
        start += step

    return children
