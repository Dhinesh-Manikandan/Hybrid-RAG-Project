"""
verify_reconciliation.py -- Diagnostic timing of rag.query() with and without warm-up.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import HybridRAG

def main():
    benchmark_file = PROJECT_ROOT / "evaluation" / "multi_document_queries.json"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    rag = HybridRAG()

    # Pre-warm
    t_warm_start = time.perf_counter()
    _ = rag.query("Warmup query")
    t_warm = time.perf_counter() - t_warm_start
    print(f"Warmup query latency: {t_warm:.2f} s")

    # Time first 10 queries
    times_15 = []
    for item in queries[:10]:
        t0 = time.perf_counter()
        _ = rag.query(item["query"], fusion_top_k=15)
        times_15.append(time.perf_counter() - t0)

    print(f"Post-warmup 10 queries (K=15) Mean: {np.mean(times_15):.2f} s, Median: {np.median(times_15):.2f} s")

    times_10 = []
    for item in queries[:10]:
        t0 = time.perf_counter()
        _ = rag.query(item["query"], fusion_top_k=10)
        times_10.append(time.perf_counter() - t0)

    print(f"Post-warmup 10 queries (K=10) Mean: {np.mean(times_10):.2f} s, Median: {np.median(times_10):.2f} s")

if __name__ == "__main__":
    main()
