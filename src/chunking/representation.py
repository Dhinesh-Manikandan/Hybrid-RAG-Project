"""
representation.py -- Canonical retrieval representation constructor.

Centralized helper ensuring consistent searchable text representation across:
- Dense vector embeddings
- BM25 lexical index
- CrossEncoder reranking

Keeps raw chunk text separate from decorated retrieval text.
Completely domain-agnostic.
"""

from pathlib import Path


def build_retrieval_text(metadata: dict, chunk_text: str = "") -> str:
    """
    Build the canonical searchable text representation for a chunk.

    Combines:
    [Document: <document_title>]
    [Section: <heading_path>]

    <chunk_text>

    Args:
        metadata:   Dict containing metadata fields (document_title, heading_path, section, source_file).
        chunk_text: The raw, unadorned chunk content. If empty, checks metadata.get("text").

    Returns:
        Searchable representation string.
    """
    parts = []

    # 1. Document Title / Root Context
    doc_title = metadata.get("document_title")
    if not doc_title and metadata.get("source_file"):
        doc_title = Path(metadata["source_file"]).stem.replace("_", " ").title()

    if doc_title:
        parts.append(f"[Document: {doc_title}]")

    # 2. Section Hierarchy Context
    heading_path = metadata.get("heading_path")
    if heading_path:
        # If heading_path contains list of headings, format hierarchy
        h_str = " > ".join(heading_path)
        parts.append(f"[Section: {h_str}]")
    elif metadata.get("section") and metadata.get("section") != "DOCUMENT":
        parts.append(f"[Section: {metadata['section']}]")

    content = chunk_text if chunk_text else metadata.get("text", "")
    prefix = "\n".join(parts)

    if prefix:
        return f"{prefix}\n\n{content}"
    return content
