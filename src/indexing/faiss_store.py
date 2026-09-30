"""
faiss_store.py -- Shared FAISS index management.

Manages three shared indexes (naive, sentence_window, parent_child),
each holding vectors from ALL documents. The document_id in metadata
maps every vector back to its source document.

Key design decisions:
- Shared indexes (not one per document) -> retrieval is always multi-document
- Duplicate prevention via chunk_id tracking
- Full rebuild is always safe and idempotent
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.config import (
    NAIVE_INDEX_PATH, NAIVE_METADATA_PATH,
    SENTENCE_WINDOW_INDEX_PATH, SENTENCE_WINDOW_META_PATH,
    PARENT_CHILD_INDEX_PATH, PARENT_CHILD_META_PATH,
    EMBEDDING_INFO_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
)


# ============================================================
# BUILD (FULL REBUILD FROM SCRATCH)
# ============================================================

def build_and_save_all_indexes(
    naive_chunks: list[dict],
    sw_chunks: list[dict],
    pc_chunks: list[dict],
    naive_embeddings: np.ndarray,
    sw_embeddings: np.ndarray,
    pc_embeddings: np.ndarray,
    model_name: str = EMBEDDING_MODEL_NAME,
    dim: int = EMBEDDING_DIM,
) -> None:
    """
    Build three FAISS indexes from scratch and save to disk.

    This is the primary build path. Called by ingest.py after
    collecting all chunks and embeddings from all documents.

    Args:
        naive_chunks:      List of naive chunk metadata dicts.
        sw_chunks:         List of sentence-window chunk metadata dicts.
        pc_chunks:         List of parent-child chunk metadata dicts.
        naive_embeddings:  Float32 array (N, dim) for naive chunks.
        sw_embeddings:     Float32 array (N, dim) for SW chunks.
        pc_embeddings:     Float32 array (N, dim) for PC chunks.
        model_name:        Embedding model used (saved to info file).
        dim:               Embedding dimension.
    """
    _save_index_and_metadata(
        naive_embeddings, naive_chunks,
        NAIVE_INDEX_PATH, NAIVE_METADATA_PATH, dim,
        name="Naive RAG",
    )

    _save_index_and_metadata(
        sw_embeddings, sw_chunks,
        SENTENCE_WINDOW_INDEX_PATH, SENTENCE_WINDOW_META_PATH, dim,
        name="Sentence Window",
    )

    _save_index_and_metadata(
        pc_embeddings, pc_chunks,
        PARENT_CHILD_INDEX_PATH, PARENT_CHILD_META_PATH, dim,
        name="Parent-Child",
    )

    # Save embedding configuration
    info = {
        "model":           model_name,
        "dimension":       dim,
        "faiss_index_type": "IndexFlatIP",
        "similarity":      "cosine (inner product on normalized vectors)",
    }
    with open(EMBEDDING_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Embedding info saved to {EMBEDDING_INFO_PATH}")


def _save_index_and_metadata(
    embeddings: np.ndarray,
    metadata: list[dict],
    index_path: Path,
    metadata_path: Path,
    dim: int,
    name: str = "",
) -> None:
    """Build a FAISS IndexFlatIP index and save index + metadata."""
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        index.add(embeddings)

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(
        f"{name}: {index.ntotal} vectors saved to {index_path}"
    )


# ============================================================
# LOAD
# ============================================================

def load_all_indexes() -> dict:
    """
    Load all three FAISS indexes and their metadata from disk.

    Returns:
        dict with keys:
            "naive":          (faiss.Index, list[dict])
            "sentence_window": (faiss.Index, list[dict])
            "parent_child":   (faiss.Index, list[dict])
            "embedding_info": dict
    """
    naive_index, naive_meta = _load_index_and_metadata(
        NAIVE_INDEX_PATH, NAIVE_METADATA_PATH, "Naive RAG"
    )

    sw_index, sw_meta = _load_index_and_metadata(
        SENTENCE_WINDOW_INDEX_PATH, SENTENCE_WINDOW_META_PATH, "Sentence Window"
    )

    pc_index, pc_meta = _load_index_and_metadata(
        PARENT_CHILD_INDEX_PATH, PARENT_CHILD_META_PATH, "Parent-Child"
    )

    embedding_info = {}
    if EMBEDDING_INFO_PATH.exists():
        with open(EMBEDDING_INFO_PATH, "r", encoding="utf-8") as f:
            embedding_info = json.load(f)

    return {
        "naive":           (naive_index, naive_meta),
        "sentence_window": (sw_index, sw_meta),
        "parent_child":    (pc_index, pc_meta),
        "embedding_info":  embedding_info,
    }


def _load_index_and_metadata(
    index_path: Path,
    metadata_path: Path,
    name: str = "",
) -> tuple:
    """Load a single FAISS index and its metadata."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"{name} index not found: {index_path}\n"
            f"Run: python ingest.py"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{name} metadata not found: {metadata_path}\n"
            f"Run: python ingest.py"
        )

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"{name}: loaded {index.ntotal} vectors")

    return index, metadata


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def get_existing_chunk_ids(metadata: list[dict]) -> set[str]:
    """
    Extract all existing chunk_ids from a metadata list.
    Used to prevent duplicate ingestion.

    Args:
        metadata: List of metadata dicts.

    Returns:
        Set of chunk_id strings.
    """
    return {m["chunk_id"] for m in metadata}


def get_existing_document_ids(metadata: list[dict]) -> set[str]:
    """
    Extract all document_ids currently in the index.

    Args:
        metadata: List of metadata dicts.

    Returns:
        Set of document_id strings.
    """
    return {m["document_id"] for m in metadata}
