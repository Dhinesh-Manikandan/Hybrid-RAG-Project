"""
sentence_window.py -- Sentence-Window chunking strategy.

For each sentence in the text, stores:
- The target sentence (used for embedding)
- A window of surrounding sentences (returned to the user)

Every representation links back to the naive chunk it was derived from
via source_chunk_id, enabling canonical RRF grouping.

Completely generic: no hardcoded section names or document-specific logic.
"""

import re
from typing import Optional


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a robust rule-based approach.
    Does not split on list numbering (e.g. '1. ', '2. ').
    """
    # (?<!\b\d) ensures we do not split on '1. ', '2. ', etc.
    raw = re.split(r"(?<!\b\d)(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def build_sentence_window_chunks(
    document_id: str,
    source_file: str,
    sections: list[dict],
    naive_chunks: list[dict],
    document_title: Optional[str] = None,
    window_size: int = 2,
) -> list[dict]:
    """
    Build sentence-window representations from section-based text.

    For each sentence:
    1. Identify its surrounding window (window_size sentences on each side).
    2. The embedding target is the window_text.
    3. source_chunk_id points to the naive chunk that covers this sentence,
       enabling canonical grouping in RRF.

    Args:
        document_id:    Stable document identifier.
        source_file:    Original filename.
        sections:       List of {"section": str, "content": str}.
        naive_chunks:   List of naive chunk dicts (for source_chunk_id lookup).
        document_title: Canonical document title.
        window_size:    Number of sentences on each side of the target.

    Returns:
        List of sentence-window chunk dicts with universal metadata schema.
    """
    from pathlib import Path
    from src.chunking.representation import build_retrieval_text

    if not document_title:
        if naive_chunks and naive_chunks[0].get("document_title"):
            document_title = naive_chunks[0]["document_title"]
        elif source_file:
            document_title = Path(source_file).stem.replace("_", " ").replace("-", " ").title()

    results = []
    global_sentence_id = 0

    # Build a lookup: character position -> naive chunk_id
    # This lets us map each sentence back to the naive chunk it belongs to
    naive_lookup = _build_naive_position_lookup(naive_chunks)

    for sec in sections:
        section_name = sec["section"]
        section_text = sec["content"]
        sentences = _split_sentences(section_text)

        heading_path = sec.get("heading_path", [section_name])

        for i, target_sentence in enumerate(sentences):
            # Build window
            window_start = max(0, i - window_size)
            window_end   = min(len(sentences), i + window_size + 1)
            window_text  = " ".join(sentences[window_start:window_end])

            # Find the nearest naive chunk for this sentence
            source_chunk_id = _find_source_chunk_id(
                target_sentence, section_name, naive_lookup, document_id, global_sentence_id
            )

            chunk_id = f"{document_id}__sw__{global_sentence_id}"

            chunk_dict = {
                "document_id":     document_id,
                "source_file":     source_file,
                "document_title":  document_title,
                "chunk_id":        chunk_id,
                "retrieval_type":  "sentence_window",
                "text":            window_text,
                "source_chunk_id": source_chunk_id,
                "parent_chunk_id": None,
                "section":         section_name,
                "heading_path":    heading_path,
                "extra": {
                    "target_sentence": target_sentence,
                    "sentence_id":     global_sentence_id,
                    "window_size":     window_size,
                },
            }
            chunk_dict["retrieval_text"] = build_retrieval_text(chunk_dict, window_text)
            results.append(chunk_dict)

            global_sentence_id += 1

    return results


def _build_naive_position_lookup(naive_chunks: list[dict]) -> list[tuple]:
    """
    Build a list of (text_prefix, chunk_id, section) tuples for matching.
    Returns list of (naive_text, chunk_id, section) tuples.
    """
    return [(c["text"], c["chunk_id"], c.get("section", "")) for c in naive_chunks]


def _find_source_chunk_id(
    target_sentence: str,
    section_name: str,
    naive_lookup: list[tuple],
    document_id: str,
    fallback_id: int,
) -> str:
    """
    Find the naive chunk_id that best corresponds to this sentence.

    Strategy:
    1. Prefer matching a naive chunk in the SAME section that contains target_sentence.
    2. Fallback to any naive chunk in the SAME section.
    3. Global fallback if section not matched.
    """
    target_clean = target_sentence.strip()

    # 1. Match within same section
    if len(target_clean) >= 10:
        for naive_text, chunk_id, sec in naive_lookup:
            if sec == section_name and target_clean in naive_text:
                return chunk_id

    for naive_text, chunk_id, sec in naive_lookup:
        if sec == section_name:
            return chunk_id

    # 2. Match across sections only if distinctive
    if len(target_clean) >= 15:
        for naive_text, chunk_id, _ in naive_lookup:
            if target_clean in naive_text:
                return chunk_id

    # Fallback: use document_id + position (still valid, just less precise)
    return f"{document_id}__naive__0"
