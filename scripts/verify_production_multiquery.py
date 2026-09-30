"""
verify_production_multiquery.py -- Validate production HybridRAG retrieval pipeline with multi-query reranking.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG

def main():
    print("=" * 80)
    print("VALIDATING PRODUCTION HYBRIDRAG WITH MULTI-QUERY RERANKING")
    print("=" * 80)

    rag = HybridRAG()

    benchmark_queries = [
        {"query": "Who is eligible for PMAY-U?",         "target": "ELIGIBILITY"},
        {"query": "What are the eligibility criteria?", "target": "ELIGIBILITY"},
        {"query": "What benefits are available?",       "target": "BENEFITS"},
        {"query": "How can beneficiaries apply?",       "target": "APPLICATION PROCESS"},
        {"query": "What documents are required?",       "target": "DOCUMENTS REQUIRED"},
    ]

    results = []
    latencies = []

    for b in benchmark_queries:
        q = b["query"]
        tgt = b["target"]

        t0 = time.perf_counter()
        res = rag.query(q, final_top_k=10)
        t_query = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_query)

        finals = res["final_results"]
        target_rank = next((idx for idx, c in enumerate(finals, start=1) if c["section"] == tgt), None)

        print("\n" + "-" * 80)
        print(f"QUERY: \"{q}\" (Target: {tgt}) | Latency: {t_query:.1f}ms")
        print(f"Target Section Rank: #{target_rank}" if target_rank else "Target Section NOT in top 10")
        print("Top 5 Results:")
        for idx, c in enumerate(finals[:5], start=1):
            sec = c.get("section", "N/A")
            score = c.get("rerank_score", 0.0)
            cid = c.get("source_chunk_id", "N/A")
            is_tgt = " <--- TARGET" if sec == tgt else ""
            print(f"  #{idx}: [{sec:<24}] Score: {score:+.4f} | Chunk: {cid}{is_tgt}")

        results.append({
            "query": q,
            "target": tgt,
            "rank": target_rank,
            "latency": t_query,
            "top5": [(c.get("section"), c.get("rerank_score")) for c in finals[:5]],
        })

    # Benchmark metrics
    n = len(benchmark_queries)
    hit1 = sum(1 for r in results if r["rank"] == 1) / n
    hit3 = sum(1 for r in results if r["rank"] is not None and r["rank"] <= 3) / n
    hit5 = sum(1 for r in results if r["rank"] is not None and r["rank"] <= 5) / n
    mrr = sum(1.0 / r["rank"] if r["rank"] is not None else 0.0 for r in results) / n
    avg_latency = sum(latencies) / n

    print("\n" + "=" * 80)
    print("PRODUCTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Hit@1       : {hit1:.2f} ({int(hit1*n)}/{n})")
    print(f"Hit@3       : {hit3:.2f} ({int(hit3*n)}/{n})")
    print(f"Hit@5       : {hit5:.2f} ({int(hit5*n)}/{n})")
    print(f"MRR         : {mrr:.4f}")
    print(f"Avg Latency : {avg_latency:.1f} ms")

if __name__ == "__main__":
    main()
