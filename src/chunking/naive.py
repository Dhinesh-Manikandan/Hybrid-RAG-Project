"""
naive.py -- Semantic Naive RAG chunking strategy.

Heading-aware, paragraph-aware, and size-bounded semantic chunking.
Completely generic: no hardcoded section names or document-specific logic.
Enforces semantic boundaries in priority order:
1. Major section boundary
2. Paragraph boundary
3. Numbered-item / bullet boundary
4. Sentence boundary
5. Hard character/token limit

Maintains raw chunk text in 'text' and canonical search representation in 'retrieval_text'.
"""

import re
from pathlib import Path
from typing import Optional
from src.config import TARGET_CHUNK_SIZE, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.cleaner import clean_text, extract_sections, extract_document_title
from src.chunking.representation import build_retrieval_text


def split_into_semantic_blocks(text: str, max_block_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """
    Split text into atomic semantic blocks (paragraphs and list items).
    Oversized blocks are split at sentence boundaries.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    blocks = []

    for p in raw_paragraphs:
        # Check if paragraph contains multiple numbered items or bullet points
        items = re.split(r"\n(?=(?:\d+[\.\)]|\([a-zA-Z0-9]+\)|[-*•])\s+)", p)
        for item in items:
            cleaned_item = item.strip()
            if cleaned_item:
                blocks.append(cleaned_item)

    final_blocks = []
    for b in blocks:
        if len(b) <= max_block_size:
            final_blocks.append(b)
        else:
            # Split oversized block at sentence boundaries
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", b) if s.strip()]
            for s in sentences:
                if len(s) <= max_block_size:
                    final_blocks.append(s)
                else:
                    # Hard slice for any extreme run-on sentence
                    start = 0
                    while start < len(s):
                        final_blocks.append(s[start:start + max_block_size].strip())
                        start += max_block_size

    return final_blocks


def build_naive_chunks_from_sections(
    document_id: str,
    source_file: str,
    sections: list[dict],
    document_title: Optional[str] = None,
    chunk_size: int = TARGET_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[dict]:
    """
    Build semantic naive chunks from a list of extracted sections.

    Each section is packed into semantic chunks respecting paragraph and item boundaries.
    Raw unadorned chunk text is stored in 'text'.
    The canonical searchable representation (document + heading context) is stored in 'retrieval_text'.

    Args:
        document_id:    Stable document identifier.
        source_file:    Original filename.
        sections:       List of {"section": str, "heading_path": list[str], "content": str}.
        document_title: Canonical document title/context (if None, derived from filename).
        chunk_size:     Target chunk size in characters (packing preference).
        chunk_overlap:  Maximum overlap between consecutive chunks of the same section.
        max_chunk_size: Hard upper limit on chunk size in characters.
        min_chunk_size: Threshold below which trailing chunks are merged with preceding ones.

    Returns:
        Flat list of chunk dicts with universal metadata schema.
    """
    if not document_title and source_file:
        document_title = Path(source_file).stem.replace("_", " ").replace("-", " ").title()

    all_chunks = []
    global_index = 0

    for sec in sections:
        section_name = sec["section"]
        heading_path = sec.get("heading_path", [section_name])
        section_text = sec.get("content", "").strip()

        if not section_text:
            continue

        blocks = split_into_semantic_blocks(section_text, max_block_size=max_chunk_size)

        sec_chunks = []
        curr_blocks = []
        curr_len = 0

        for block in blocks:
            block_len = len(block)
            # Would adding this block exceed MAX_CHUNK_SIZE?
            if curr_blocks and (curr_len + block_len + 2 > max_chunk_size):
                sec_chunks.append("\n\n".join(curr_blocks))
                # Check for small overlap carryover
                overlap_block = curr_blocks[-1]
                if len(overlap_block) <= chunk_overlap and len(overlap_block) + block_len + 2 <= max_chunk_size:
                    curr_blocks = [overlap_block, block]
                    curr_len = len(overlap_block) + block_len + 2
                else:
                    curr_blocks = [block]
                    curr_len = block_len
            elif curr_blocks and (curr_len >= chunk_size):
                # We reached preferred target size at a clean semantic boundary
                sec_chunks.append("\n\n".join(curr_blocks))
                curr_blocks = [block]
                curr_len = block_len
            else:
                curr_blocks.append(block)
                curr_len += block_len + 2

        if curr_blocks:
            # Check if trailing small block can merge into previous chunk
            if sec_chunks and (curr_len < min_chunk_size) and (len(sec_chunks[-1]) + curr_len + 2 <= max_chunk_size):
                sec_chunks[-1] += "\n\n" + "\n\n".join(curr_blocks)
            else:
                sec_chunks.append("\n\n".join(curr_blocks))

        for section_index, chunk_content in enumerate(sec_chunks):
            raw_text = chunk_content.strip()
            chunk_id = f"{document_id}__naive__{global_index}"

            chunk_dict = {
                "document_id":     document_id,
                "source_file":     source_file,
                "document_title":  document_title,
                "chunk_id":        chunk_id,
                "retrieval_type":  "naive",
                "text":            raw_text,  # Clean, unadorned text for final answer generation
                "source_chunk_id": chunk_id,
                "parent_chunk_id": None,
                "section":         section_name,
                "heading_path":    heading_path,
                "extra": {
                    "section_index": section_index,
                    "global_index":  global_index,
                },
            }
            # Searchable representation for embeddings, BM25, and reranker
            chunk_dict["retrieval_text"] = build_retrieval_text(chunk_dict, raw_text)

            all_chunks.append(chunk_dict)
            global_index += 1

    return all_chunks


def build_naive_chunks(
    document_id: str,
    source_file: str,
    text: str,
    document_title: Optional[str] = None,
    section: str = "DOCUMENT",
    chunk_size: int = TARGET_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into semantic chunks.
    Automatically runs section extraction and title extraction to preserve document structure.
    """
    clean = clean_text(text)
    if not document_title:
        document_title = extract_document_title(clean, source_file)
    sections = extract_sections(clean)
    return build_naive_chunks_from_sections(
        document_id=document_id,
        source_file=source_file,
        sections=sections,
        document_title=document_title,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
