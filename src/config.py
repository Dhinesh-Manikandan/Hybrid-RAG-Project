from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
VECTOR_DIR     = DATA_DIR / "vector_store"
RETRIEVAL_DIR  = DATA_DIR / "retrieval"
EVAL_DIR       = PROJECT_DIR / "evaluation"

for _d in [RAW_DIR, PROCESSED_DIR, VECTOR_DIR, RETRIEVAL_DIR, EVAL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TARGET_CHUNK_SIZE = 800
MAX_CHUNK_SIZE    = 1200
MIN_CHUNK_SIZE    = 150
CHUNK_OVERLAP     = 128

# Kept for backwards compatibility
NAIVE_CHUNK_SIZE    = TARGET_CHUNK_SIZE
NAIVE_CHUNK_OVERLAP = CHUNK_OVERLAP

SENTENCE_WINDOW_SIZE = 2

PARENT_CHUNK_SIZE   = 512
CHILD_CHUNK_SIZE    = 128
CHILD_CHUNK_OVERLAP = 16

RETRIEVAL_TOP_K = 10

RRF_K = 60

# Candidate pool size after RRF fusion
# Baseline: FUSION_TOP_K = 15
FUSION_TOP_K = 15
MMR_LAMBDA   = 0.7

FINAL_TOP_K = 5

NAIVE_INDEX_PATH           = VECTOR_DIR / "naive_rag.index"
NAIVE_METADATA_PATH        = VECTOR_DIR / "naive_rag_metadata.pkl"

SENTENCE_WINDOW_INDEX_PATH = VECTOR_DIR / "sentence_window.index"
SENTENCE_WINDOW_META_PATH  = VECTOR_DIR / "sentence_window_metadata.pkl"

PARENT_CHILD_INDEX_PATH    = VECTOR_DIR / "parent_child.index"
PARENT_CHILD_META_PATH     = VECTOR_DIR / "parent_child_metadata.pkl"

BM25_CORPUS_PATH    = VECTOR_DIR / "bm25_corpus.pkl"
EMBEDDING_INFO_PATH = VECTOR_DIR / "embedding_info.json"

GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"

# Query Reformulation & Multi-Query Retrieval
ENABLE_QUERY_REFORMULATION = True
# Total query formulations / perspectives evaluated (V=3: 1 original + 2 linguistic reformulations)
# Baseline: NUM_REFORMULATIONS = 3 (V=3 query perspectives)
NUM_REFORMULATIONS         = 3
REFORMULATION_METHOD       = "linguistic"  # Options: "linguistic", "model", "none"
REFORMULATION_MODEL_NAME   = "google/flan-t5-small"

# Multi-Query CrossEncoder Reranking
ENABLE_MULTI_QUERY_RERANKING = True

# CrossEncoder Batch Size
# Baseline: BATCH_SIZE = 32
BATCH_SIZE                   = 32
CROSS_ENCODER_BATCH_SIZE     = BATCH_SIZE

# LLM Answer Generation
LLM_MODEL_NAME           = "Qwen/Qwen3-0.6B"
LLM_TEMPERATURE          = 0.1
LLM_MAX_NEW_TOKENS       = 512
LLM_ENABLE_THINKING      = False
GENERATION_CONTEXT_TOP_K = 5

# Relevance gate: if the best CrossEncoder rerank_score for the retrieved passages
# is below this threshold, the query is considered out-of-scope and the LLM is
# skipped — returning a clean "not in context" response instead of hallucinating.
# Calibrated on ms-marco-MiniLM-L-6-v2 raw logit scores:
#   unrelated queries  → ~-11 to -12   (below threshold → gated out)
#   scheme queries     → ~-9  to -10   (above threshold → passed through)
RERANK_RELEVANCE_THRESHOLD = -10.5


