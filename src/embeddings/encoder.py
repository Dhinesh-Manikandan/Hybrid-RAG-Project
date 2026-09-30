"""
encoder.py -- Text embedding utilities.

Wraps SentenceTransformer with consistent normalization and batching.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    Args:
        model_name: HuggingFace model name or local path.

    Returns:
        Loaded SentenceTransformer instance.
    """
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into normalized embeddings.

    Args:
        texts:         List of text strings to encode.
        model_name:    Embedding model name.
        batch_size:    Batch size for encoding.
        show_progress: Whether to show a progress bar.

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dim).
    """
    model = get_model(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )

    return np.asarray(embeddings, dtype="float32")


def encode_query(query: str, model_name: str) -> np.ndarray:
    """
    Encode a single query string into a normalized embedding.

    Args:
        query:      Query string.
        model_name: Embedding model name.

    Returns:
        Float32 numpy array of shape (1, embedding_dim).
    """
    model = get_model(model_name)

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    return np.asarray(embedding, dtype="float32")
