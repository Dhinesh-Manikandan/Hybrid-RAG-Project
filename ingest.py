"""
ingest.py -- Universal document ingestion and indexing pipeline.

Usage:
    python ingest.py                        # Process all docs in data/raw/
    python ingest.py --rebuild              # Force full rebuild (default)
    python ingest.py --doc path/to/doc.txt  # Process a single document

Adding a new document:
    1. Copy document to data/raw/
    2. Run: python ingest.py
    3. Done. Retrieval automatically covers the new document.

No modification to retrieval, fusion, or reranking code is needed.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from src.config import (
    RAW_DIR, PROCESSED_DIR, VECTOR_DIR,
    SUPPORTED_EXTENSIONS,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    NAIVE_CHUNK_SIZE, NAIVE_CHUNK_OVERLAP,
    SENTENCE_WINDOW_SIZE,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    BM25_CORPUS_PATH,
)
from src.ingestion.document_id  import generate_document_id
from src.ingestion.loader       import load_document
from src.ingestion.cleaner      import clean_text, extract_sections, extract_document_title
from src.chunking.naive         import build_naive_chunks_from_sections
from src.chunking.sentence_window import build_sentence_window_chunks
from src.chunking.parent_child  import build_parent_child_chunks
from src.embeddings.encoder     import encode_texts
from src.indexing.faiss_store   import build_and_save_all_indexes
from src.retrieval.bm25         import BM25Retriever
from src.utils.io               import save_json


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def discover_documents(raw_dir: Path) -> list[Path]:
    """
    Discover all supported documents in the raw data directory.

    Args:
        raw_dir: Path to data/raw/

    Returns:
        List of file paths with supported extensions.
    """
    docs = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(path)

    print(f"\nDiscovered {len(docs)} document(s) in {raw_dir}:")
    for d in docs:
        print(f"  - {d.name}")

    return docs


# ============================================================
# SINGLE DOCUMENT PROCESSING
# ============================================================

def process_document(path: Path) -> dict:
    """
    Process a single document into all representations.

    Returns a dict with:
        document_id, source_file,
        naive_chunks, sw_chunks, pc_chunks,
        processed_metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {path.name}")
    print(f"{'='*60}")

    # 1. Generate stable document_id
    document_id = generate_document_id(path)
    source_file = path.name
    print(f"document_id: {document_id}")

    # 2. Load raw text
    raw_text = load_document(path)
    print(f"Loaded: {len(raw_text):,} characters")

    # 3. Clean text
    clean = clean_text(raw_text)
    print(f"Cleaned: {len(clean):,} characters")

    # 4. Extract canonical document title
    document_title = extract_document_title(clean, source_file)
    print(f"document_title: {document_title}")

    # 5. Extract sections (generic heuristic, no hardcoded section names)
    sections = extract_sections(clean)
    print(f"Sections detected: {len(sections)}")
    for s in sections:
        print(f"  [{s['section']}]: {len(s['content'])} chars")

    # 6. Build naive chunks (canonical base unit)
    naive_chunks = build_naive_chunks_from_sections(
        document_id=document_id,
        source_file=source_file,
        sections=sections,
        document_title=document_title,
        chunk_size=NAIVE_CHUNK_SIZE,
        chunk_overlap=NAIVE_CHUNK_OVERLAP,
    )
    print(f"Naive chunks: {len(naive_chunks)}")

    # 7. Build sentence-window chunks
    sw_chunks = build_sentence_window_chunks(
        document_id=document_id,
        source_file=source_file,
        sections=sections,
        naive_chunks=naive_chunks,
        document_title=document_title,
        window_size=SENTENCE_WINDOW_SIZE,
    )
    print(f"Sentence-window chunks: {len(sw_chunks)}")

    # 8. Build parent-child chunks
    pc_chunks = build_parent_child_chunks(
        document_id=document_id,
        source_file=source_file,
        naive_chunks=naive_chunks,
        document_title=document_title,
        child_chunk_size=CHILD_CHUNK_SIZE,
        child_chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
    print(f"Parent-child chunks: {len(pc_chunks)}")

    # 9. Save processed metadata (for inspection/debugging)
    processed_meta = {
        "document_id":    document_id,
        "source_file":    source_file,
        "document_title": document_title,
        "n_sections":     len(sections),
        "n_naive":        len(naive_chunks),
        "n_sw":           len(sw_chunks),
        "n_pc":           len(pc_chunks),
        "sections":       [s["section"] for s in sections],
    }
    processed_path = PROCESSED_DIR / f"{path.stem}_processed.json"
    save_json(processed_meta, processed_path)
    print(f"Processed metadata: {processed_path}")

    return {
        "document_id":  document_id,
        "source_file":  source_file,
        "naive_chunks": naive_chunks,
        "sw_chunks":    sw_chunks,
        "pc_chunks":    pc_chunks,
        "metadata":     processed_meta,
    }


# ============================================================
# FULL PIPELINE
# ============================================================

def run_ingestion(doc_paths: list[Path]) -> None:
    """
    Run the full ingestion pipeline for a list of documents.

    1. Process each document into chunks.
    2. Collect all chunks across all documents.
    3. Generate embeddings.
    4. Build and save shared FAISS indexes.
    5. Build and save BM25 index.
    """
    if not doc_paths:
        print("No documents found to process.")
        return

    # Aggregate chunks from all documents
    all_naive_chunks = []
    all_sw_chunks    = []
    all_pc_chunks    = []

    for path in doc_paths:
        result = process_document(path)
        all_naive_chunks.extend(result["naive_chunks"])
        all_sw_chunks.extend(result["sw_chunks"])
        all_pc_chunks.extend(result["pc_chunks"])

    print(f"\n{'='*60}")
    print("TOTALS ACROSS ALL DOCUMENTS")
    print(f"{'='*60}")
    print(f"Total naive chunks:          {len(all_naive_chunks)}")
    print(f"Total sentence-window chunks: {len(all_sw_chunks)}")
    print(f"Total parent-child chunks:    {len(all_pc_chunks)}")

    # Generate embeddings using canonical retrieval representations
    print(f"\nGenerating embeddings with: {EMBEDDING_MODEL_NAME}")

    print("\nEncoding naive chunks (using retrieval_text)...")
    naive_embeddings = encode_texts(
        [c.get("retrieval_text", c["text"]) for c in all_naive_chunks],
        EMBEDDING_MODEL_NAME,
    )

    print("\nEncoding sentence-window chunks (using retrieval_text)...")
    sw_embeddings = encode_texts(
        [c.get("retrieval_text", c["text"]) for c in all_sw_chunks],
        EMBEDDING_MODEL_NAME,
    )

    print("\nEncoding parent-child chunks (using retrieval_text)...")
    pc_embeddings = encode_texts(
        [c.get("retrieval_text", c["text"]) for c in all_pc_chunks],
        EMBEDDING_MODEL_NAME,
    )

    print(f"\nNaive embeddings shape:          {naive_embeddings.shape}")
    print(f"Sentence-window embeddings shape: {sw_embeddings.shape}")
    print(f"Parent-child embeddings shape:    {pc_embeddings.shape}")

    # Build and save FAISS indexes
    print("\nBuilding FAISS indexes...")
    build_and_save_all_indexes(
        naive_chunks=all_naive_chunks,
        sw_chunks=all_sw_chunks,
        pc_chunks=all_pc_chunks,
        naive_embeddings=naive_embeddings,
        sw_embeddings=sw_embeddings,
        pc_embeddings=pc_embeddings,
        model_name=EMBEDDING_MODEL_NAME,
        dim=EMBEDDING_DIM,
    )

    # Build and save BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever()
    bm25.build(all_naive_chunks)
    bm25.save(BM25_CORPUS_PATH)

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Documents processed: {len(doc_paths)}")
    print(f"Vector store:        {VECTOR_DIR}")
    print(f"BM25 corpus:         {BM25_CORPUS_PATH}")
    print(f"\nReady for retrieval. Run your notebooks or query pipeline.")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid-RAG ingestion pipeline. "
                    "Processes all documents in data/raw/ and builds shared indexes."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Process a single document (path). "
             "If not specified, processes all documents in data/raw/."
    )
    args = parser.parse_args()

    if args.doc:
        doc_path = Path(args.doc)
        if not doc_path.exists():
            print(f"Error: Document not found: {doc_path}")
            sys.exit(1)
        doc_paths = [doc_path]
    else:
        doc_paths = discover_documents(RAW_DIR)

    if not doc_paths:
        print(f"\nNo supported documents found in {RAW_DIR}")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        print(f"Add documents to {RAW_DIR} and re-run.")
        sys.exit(0)

    run_ingestion(doc_paths)


if __name__ == "__main__":
    main()
