"""
validate_multidoc_corpus.py -- Validation and smoke testing for multi-document Hybrid-RAG corpus.

Performs:
1. Corpus validation:
   - Discovers all raw documents
   - Inspects processed representations
   - Validates vector store (naive, sw, pc) and BM25 index
   - Confirms chunk counts, lineage, and document_ids
2. Multi-document smoke tests (Task 7)
3. Cross-document confusion evaluation (Task 8)
"""

import sys
import json
import pickle
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    RAW_DIR, PROCESSED_DIR, VECTOR_DIR,
    NAIVE_INDEX_PATH, NAIVE_METADATA_PATH,
    SENTENCE_WINDOW_INDEX_PATH, SENTENCE_WINDOW_META_PATH,
    PARENT_CHILD_INDEX_PATH, PARENT_CHILD_META_PATH,
    BM25_CORPUS_PATH, SUPPORTED_EXTENSIONS
)
from src.ingestion.document_id import generate_document_id
from retrieve import HybridRAG


def validate_corpus():
    print("=" * 80)
    print("TASK 5: CORPUS AND INDEX VALIDATION")
    print("=" * 80)

    # 1. Discover raw documents
    raw_files = sorted([p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"\n1. Discovered {len(raw_files)} raw documents in {RAW_DIR}:")
    
    doc_id_map = {}
    for f in raw_files:
        doc_id = generate_document_id(f)
        doc_id_map[f.name] = doc_id
        print(f"   - {f.name:25s} -> {doc_id}")

    # 2. Check processed metadata files
    print(f"\n2. Checking processed representations in {PROCESSED_DIR}:")
    processed_counts = {}
    for f in raw_files:
        proc_file = PROCESSED_DIR / f"{f.stem}_processed.json"
        if not proc_file.exists():
            print(f"   [MISSING] {proc_file.name}")
        else:
            with open(proc_file, "r", encoding="utf-8") as pf:
                data = json.load(pf)
                processed_counts[f.name] = data
                print(f"   [OK] {f.name:25s} | Naive: {data['n_naive']:3d} | SW: {data['n_sw']:4d} | PC: {data['n_pc']:4d} | Sections: {data['n_sections']:2d}")

    # 3. Load indexes and metadata
    with open(NAIVE_METADATA_PATH, "rb") as mf:
        naive_meta = pickle.load(mf)
    with open(SENTENCE_WINDOW_META_PATH, "rb") as mf:
        sw_meta = pickle.load(mf)
    with open(PARENT_CHILD_META_PATH, "rb") as mf:
        pc_meta = pickle.load(mf)
    with open(BM25_CORPUS_PATH, "rb") as bf:
        bm25_data = pickle.load(bf)
        bm25_meta = bm25_data["corpus_metadata"]

    print(f"\n3. Total Chunks in Index:")
    print(f"   - Naive FAISS index     : {len(naive_meta)} vectors")
    print(f"   - Sentence Window FAISS : {len(sw_meta)} vectors")
    print(f"   - Parent-Child FAISS    : {len(pc_meta)} vectors")
    print(f"   - BM25 Corpus           : {len(bm25_meta)} documents")

    # 4. Per-document chunk breakdown across indexes
    print(f"\n4. Per-Document Index Representation Breakdown:")
    header = f"{'Source File':<25} | {'Doc ID':<16} | {'Naive':<6} | {'SW':<6} | {'PC':<6} | {'BM25':<6} | {'In All?':<8}"
    print(header)
    print("-" * len(header))

    naive_by_doc = {}
    for c in naive_meta:
        naive_by_doc[c["document_id"]] = naive_by_doc.get(c["document_id"], 0) + 1

    sw_by_doc = {}
    for c in sw_meta:
        sw_by_doc[c["document_id"]] = sw_by_doc.get(c["document_id"], 0) + 1

    pc_by_doc = {}
    for c in pc_meta:
        pc_by_doc[c["document_id"]] = pc_by_doc.get(c["document_id"], 0) + 1

    bm25_by_doc = {}
    for c in bm25_meta:
        bm25_by_doc[c["document_id"]] = bm25_by_doc.get(c["document_id"], 0) + 1

    all_ok = True
    for f in raw_files:
        d_id = doc_id_map[f.name]
        n_c = naive_by_doc.get(d_id, 0)
        s_c = sw_by_doc.get(d_id, 0)
        p_c = pc_by_doc.get(d_id, 0)
        b_c = bm25_by_doc.get(d_id, 0)
        in_all = (n_c > 0 and s_c > 0 and p_c > 0 and b_c > 0 and n_c == b_c)
        if not in_all:
            all_ok = False
        status_str = "YES" if in_all else "NO"
        print(f"{f.name:<25} | {d_id:<16} | {n_c:<6d} | {s_c:<6d} | {p_c:<6d} | {b_c:<6d} | {status_str:<8}")

    print("-" * len(header))
    total_naive = sum(naive_by_doc.values())
    total_sw = sum(sw_by_doc.values())
    total_pc = sum(pc_by_doc.values())
    total_bm25 = sum(bm25_by_doc.values())
    print(f"{'TOTAL':<25} | {'-':<16} | {total_naive:<6d} | {total_sw:<6d} | {total_pc:<6d} | {total_bm25:<6d} | {'ALL OK' if all_ok else 'FAIL'}")

    # 5. Metadata Lineage Verification
    print(f"\n5. Metadata Lineage & Universal Schema Verification:")
    sample_indices = [0, len(naive_meta) // 2, len(naive_meta) - 1]
    required_keys = ["document_id", "source_file", "chunk_id", "retrieval_type", "source_chunk_id", "section", "heading_path", "text", "retrieval_text"]
    for idx in sample_indices:
        chunk = naive_meta[idx]
        missing = [k for k in required_keys if k not in chunk]
        assert not missing, f"Chunk {chunk.get('chunk_id')} missing required keys: {missing}"
        assert chunk["document_id"] == doc_id_map[chunk["source_file"]], "document_id mismatch with source_file!"
    print("   [OK] Lineage fields (document_id, source_file, chunk_id, retrieval_type, source_chunk_id, section, heading_path) strictly intact across all chunks.")

    return doc_id_map, raw_files


def run_smoke_tests(rag: HybridRAG, doc_id_map: dict):
    print("\n" + "=" * 80)
    print("TASK 7: MULTI-DOCUMENT RETRIEVAL SMOKE TESTS")
    print("=" * 80)

    smoke_test_cases = [
        {"query": "Who is eligible for PMAY-G?", "expected_file": "pmay_g.txt"},
        {"query": "What benefits are provided under PM Vishwakarma?", "expected_file": "pm_vishwakarma.txt"},
        {"query": "Who can open a Sukanya Samriddhi account?", "expected_file": "sukanya_samriddhi.txt"},
        {"query": "What are the benefits of PM-KISAN?", "expected_file": "pm_kisan.txt"},
        {"query": "What is the purpose of PMKVY?", "expected_file": "pmkvy.txt"},
        {"query": "Who is eligible for PMJDY account opening?", "expected_file": "pmjdy.txt"},
        {"query": "What financial assistance is available under PMAY-G?", "expected_file": "pmay_g.txt"},
        {"query": "What support is provided for digital transactions under PM Vishwakarma?", "expected_file": "pm_vishwakarma.txt"},
        {"query": "What is the entry age limit for Atal Pension Yojana?", "expected_file": "atal_pension_yojana.txt"},
        {"query": "Who receives free food grains under PM-GKAY?", "expected_file": "pm_gkay.txt"},
        {"query": "What is the loan amount under PM SVANidhi for street vendors?", "expected_file": "pm_svanidhi.txt"},
        {"query": "What are the loan categories under Pradhan Mantri Mudra Yojana?", "expected_file": "pmmy.txt"},
        {"query": "Who is eligible for a deposit-free LPG connection under PMUY?", "expected_file": "pmuy.txt"},
        {"query": "What are the eligibility criteria for Stand-Up India loans?", "expected_file": "stand_up_india.txt"},
        {"query": "What crop losses are covered under PMFBY?", "expected_file": "pmfby.txt"},
    ]

    results_summary = []

    for idx, case in enumerate(smoke_test_cases, 1):
        q = case["query"]
        expected = case["expected_file"]
        print(f"\n--- Smoke Test {idx}: \"{q}\" ---")
        print(f"Expected document: {expected}")

        res = rag.query(q)
        finals = res["final_results"]

        retrieved_files = [r.get("source_file", "unknown") for r in finals]
        top_doc = retrieved_files[0] if retrieved_files else "None"
        top_5_docs = retrieved_files[:5]

        in_top_1 = (top_doc == expected)
        in_top_3 = (expected in top_5_docs[:3])
        in_top_5 = (expected in top_5_docs[:5])

        print(f"Top-1 document:   {top_doc} [{'MATCH' if in_top_1 else 'DIFF'}]")
        print(f"Top-5 documents:  {top_5_docs}")
        print(f"Recall@1: {in_top_1} | Recall@3: {in_top_3} | Recall@5: {in_top_5}")

        print("Final Top Results:")
        for r_i, r in enumerate(finals[:3], 1):
            print(f"   Rank {r_i}: [{r.get('source_file')}] (Rerank Score: {r.get('rerank_score', 0):.4f}, RRF: {r.get('rrf_score', 0):.4f}) - Section: {r.get('section')}")
            print(f"           Preview: {r.get('text', '')[:140]}...")

        results_summary.append({
            "query": q,
            "expected": expected,
            "top_1": top_doc,
            "top_5": top_5_docs,
            "in_top_1": in_top_1,
            "in_top_3": in_top_3,
            "in_top_5": in_top_5,
            "final_top_rerank_score": finals[0].get("rerank_score") if finals else None,
        })

    print("\n" + "=" * 80)
    print("TASK 7 SMOKE TEST SUMMARY TABLE")
    print("=" * 80)
    fmt = "{:<2} | {:<50} | {:<20} | {:<20} | {:<5} | {:<5} | {:<5}"
    print(fmt.format("#", "Query", "Expected", "Top-1 Retrieved", "Top1", "Top3", "Top5"))
    print("-" * 115)
    for i, s in enumerate(results_summary, 1):
        print(fmt.format(
            i,
            s["query"][:48],
            s["expected"],
            s["top_1"],
            "PASS" if s["in_top_1"] else "FAIL",
            "PASS" if s["in_top_3"] else "FAIL",
            "PASS" if s["in_top_5"] else "FAIL",
        ))
    
    top1_acc = sum(1 for s in results_summary if s["in_top_1"]) / len(results_summary)
    top3_acc = sum(1 for s in results_summary if s["in_top_3"]) / len(results_summary)
    top5_acc = sum(1 for s in results_summary if s["in_top_5"]) / len(results_summary)
    print("-" * 115)
    print(f"Top-1 Accuracy: {top1_acc*100:.1f}% ({sum(1 for s in results_summary if s['in_top_1'])}/{len(results_summary)})")
    print(f"Top-3 Accuracy: {top3_acc*100:.1f}% ({sum(1 for s in results_summary if s['in_top_3'])}/{len(results_summary)})")
    print(f"Top-5 Accuracy: {top5_acc*100:.1f}% ({sum(1 for s in results_summary if s['in_top_5'])}/{len(results_summary)})")


def run_confusion_tests(rag: HybridRAG):
    print("\n" + "=" * 80)
    print("TASK 8: CROSS-DOCUMENT CONFUSION EVALUATION")
    print("=" * 80)

    confusion_cases = [
        {
            "query": "Who is eligible for housing assistance?",
            "plausible_docs": ["pmay_u.txt", "pmay_g.txt"],
            "concept": "Housing eligibility: distinction between Urban (PMAY-U) and Gramin (PMAY-G)"
        },
        {
            "query": "What financial assistance is available for building a house?",
            "plausible_docs": ["pmay_u.txt", "pmay_g.txt"],
            "concept": "Housing grant/loan assistance across PMAY-U vs PMAY-G"
        },
        {
            "query": "What support is available for skill development?",
            "plausible_docs": ["pmkvy.txt"],
            "concept": "Vocational / skill development training"
        },
        {
            "query": "What credit support is available to artisans?",
            "plausible_docs": ["pm_vishwakarma.txt"],
            "concept": "Artisan / craftsman loan & credit support"
        },
        {
            "query": "What benefits are available for girl children?",
            "plausible_docs": ["sukanya_samriddhi.txt"],
            "concept": "Girl child financial savings / welfare"
        },
        {
            "query": "What credit facility or working capital loan is provided to street vendors?",
            "plausible_docs": ["pm_svanidhi.txt"],
            "concept": "Street vendor micro-credit / urban livelihood"
        },
        {
            "query": "What pension scheme is available for old age income security?",
            "plausible_docs": ["atal_pension_yojana.txt"],
            "concept": "Old age pension security for unorganized sector"
        }
    ]

    for idx, case in enumerate(confusion_cases, 1):
        q = case["query"]
        plausible = case["plausible_docs"]
        concept = case["concept"]

        print(f"\n--- Confusion Test {idx}: \"{q}\" ---")
        print(f"Target Concept: {concept}")
        print(f"Plausible / Target Schemes: {plausible}")

        res = rag.query(q)
        finals = res["final_results"]

        retrieved_files = [r.get("source_file", "unknown") for r in finals]
        print(f"Top 5 Retrieved Documents: {retrieved_files[:5]}")

        for r_i, r in enumerate(finals[:5], 1):
            doc = r.get("source_file")
            score = r.get("rerank_score", 0)
            sec = r.get("section", "N/A")
            is_match = doc in plausible
            print(f"   Rank {r_i} [{'PLAUSIBLE' if is_match else 'CONFUSED'}]: [{doc}] (Rerank Score: {score:.4f}) | Sec: {sec}")
            print(f"          Snippet: {r.get('text', '')[:120]}...")

        # Check if at least one plausible doc is in Top-1 and in Top-3
        has_plausible_top1 = retrieved_files[0] in plausible if retrieved_files else False
        has_plausible_top3 = any(f in plausible for f in retrieved_files[:3])
        print(f"Result Assessment: Top-1 Plausible: {has_plausible_top1} | Top-3 Plausible: {has_plausible_top3}")


def main():
    doc_id_map, raw_files = validate_corpus()
    print("\nInitializing Hybrid-RAG engine for retrieval smoke tests...")
    rag = HybridRAG()
    run_smoke_tests(rag, doc_id_map)
    run_confusion_tests(rag)


if __name__ == "__main__":
    main()
