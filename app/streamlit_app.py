import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import HybridRAG
from src.config import FUSION_TOP_K, FINAL_TOP_K

st.set_page_config(page_title="Hybrid-RAG Research Dashboard", layout="wide", page_icon="🔬")

# Initialize session state for chat BEFORE css so we can conditionally style
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Custom CSS for polished look & precision alignment
st.markdown(f"""
    <style>
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 4px; padding-top: 10px; padding-bottom: 10px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; }}
        h1, h2, h3 {{ color: inherit; }}
        
        /* Centered ChatGPT-like conversation layout */
        section[data-testid="stMain"] .block-container {{
            padding-bottom: 120px !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }}

        /* Dynamic chat input placement */
        [data-testid="stChatInput"] > div {{
            width: 100% !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }}
        [data-testid="stChatInput"] textarea {{
            border-radius: 12px !important;
        }}

        /* Suggestion Prompt Cards */
        div[data-testid="column"] .stButton > button {{
            height: auto !important;
            min-height: 74px !important;
            padding: 14px 18px !important;
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #F8FAFC !important;
            text-align: left !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        div[data-testid="column"] .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.07) !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
        }}

        div[data-testid="column"] .stButton > button:active {{
            transform: translateY(0px) !important;
        }}

        div[data-testid="column"] .stButton > button p {{
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
            font-size: 0.92rem !important;
            line-height: 1.45 !important;
        }}

        /* Sidebar New Chat Button styling */
        [data-testid="stSidebar"] button[kind="primary"] {{
            background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 10px 16px !important;
            transition: all 0.2s ease !important;
        }}
        [data-testid="stSidebar"] button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* ------------------------------------------------------------------ */
        /* Exact Avatar & Container Precision Alignment                       */
        /* ------------------------------------------------------------------ */
        [data-testid="stChatMessage"] {{
            display: flex !important;
            flex-direction: row !important;
            align-items: flex-start !important;
            gap: 14px !important;
            padding: 12px 16px !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}

        /* User message container */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            background-color: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
        }}

        /* Assistant message container */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
            background-color: transparent !important;
            border: 1px solid transparent !important;
        }}

        /* Both avatars aligned to exact identical dimensions and position */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarCustom"] {{
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            min-height: 32px !important;
            max-width: 32px !important;
            max-height: 32px !important;
            border-radius: 8px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
        }}

        /* Ensure material icons inside avatars are centered */
        [data-testid="stChatMessageAvatarUser"] span,
        [data-testid="stChatMessageAvatarAssistant"] span {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 20px !important;
            line-height: 1 !important;
        }}

        /* Message content wrapper */
        [data-testid="stChatMessageContent"] {{
            flex-grow: 1 !important;
            min-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }}

        /* User message text aligned and centered to the avatar horizontal axis */
        .user-message-text {{
            display: flex !important;
            align-items: center !important;
            min-height: 32px !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            color: #F8FAFC !important;
        }}

        /* Fallback for standard user message paragraph */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div:first-child > p:first-child {{
            min-height: 32px !important;
            display: flex !important;
            align-items: center !important;
            margin: 0 !important;
        }}

        /* AI Answer Header - centered with orange emoji on the EXACT horizontal axis */
        .ai-answer-header {{
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            height: 32px !important;
            min-height: 32px !important;
            max-height: 32px !important;
            margin-top: 0px !important;
            margin-bottom: 8px !important;
            padding: 0px !important;
            line-height: 32px !important;
        }}

        .ai-answer-header .ai-badge-icon {{
            font-size: 1.2rem !important;
            line-height: 1 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        .ai-answer-header .ai-answer-title {{
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            line-height: 1 !important;
            display: inline-flex !important;
            align-items: center !important;
            letter-spacing: -0.01em !important;
        }}

        /* Global reset for h3 inside chat to prevent pushing down */
        [data-testid="stChatMessageContent"] h3:first-child {{
            margin-top: 0 !important;
            height: 32px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }}

        /* ChatGPT-style thinking status widget */
        [data-testid="stStatusWidget"] {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            margin-top: 4px !important;
            margin-bottom: 12px !important;
            padding: 4px 12px !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stStatusWidget"]:hover {{
            border-color: rgba(255, 255, 255, 0.2) !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }}

        .thought-step-item {{
            font-size: 0.9rem !important;
            color: #94A3B8 !important;
            padding: 2px 0 !important;
            display: flex !important;
            align-items: center !important;
            gap: 6px !important;
        }}
        
        /* ------------------------------------------------------------------ */
        /* Lock Chat Input Button & Tooltip While Query is Processing          */
        /* ------------------------------------------------------------------ */
        body:has([data-testid="stStatusWidgetRunningIcon"]) [data-testid="stChatInputSubmitButton"],
        body:has([data-testid="stStatusWidgetRunningManIcon"]) [data-testid="stChatInputSubmitButton"],
        body:has([aria-label="Running..."]) [data-testid="stChatInputSubmitButton"],
        body:has([data-testid="stStatusWidgetRunningIcon"]) [data-testid="stChatInput"] button,
        body:has([data-testid="stStatusWidgetRunningManIcon"]) [data-testid="stChatInput"] button,
        body:has([aria-label="Running..."]) [data-testid="stChatInput"] button,
        [data-testid="stChatInput"].is-processing [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInput"].is-processing button {{
            pointer-events: none !important;
            opacity: 0.3 !important;
            filter: grayscale(1) !important;
            cursor: not-allowed !important;
        }}

        body:has([data-testid="stStatusWidgetRunningIcon"]) [data-testid="stChatInput"] textarea,
        body:has([data-testid="stStatusWidgetRunningManIcon"]) [data-testid="stChatInput"] textarea,
        body:has([aria-label="Running..."]) [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"].is-processing textarea {{
            cursor: not-allowed !important;
        }}

        /* Tooltip when hovering over input area while query is running */
        body:has([data-testid="stStatusWidgetRunningIcon"]) [data-testid="stChatInput"]:hover::after,
        body:has([data-testid="stStatusWidgetRunningManIcon"]) [data-testid="stChatInput"]:hover::after,
        body:has([aria-label="Running..."]) [data-testid="stChatInput"]:hover::after,
        [data-testid="stChatInput"].is-processing:hover::after {{
            content: "⏳ Already a query is running. Please wait...";
            position: absolute;
            bottom: calc(100% + 10px);
            right: 20px;
            background: #0F172A;
            color: #EF4444;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.4);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
            pointer-events: none;
            z-index: 999999;
            white-space: nowrap;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing frozen backend models & indexes (this happens only once)...")
def load_rag():
    import importlib
    import src.config as cfg
    import src.generation.context_builder as ctx_builder
    import src.generation.llm_generator as llm_gen
    import retrieve
    importlib.reload(cfg)
    importlib.reload(ctx_builder)
    importlib.reload(llm_gen)
    importlib.reload(retrieve)

    from retrieve import HybridRAG
    rag = HybridRAG()
    # Force generator initialization to load the model into memory
    rag.get_generator()
    return rag

@st.cache_resource
def get_shared_evaluator_model():
    from sentence_transformers import SentenceTransformer
    import src.config as cfg
    return SentenceTransformer(cfg.EMBEDDING_MODEL_NAME)

def load_evaluator():
    import importlib
    import src.evaluation.simple_evaluator as simple_eval_module
    importlib.reload(simple_eval_module)
    evaluator = simple_eval_module.SimpleEvaluator()
    evaluator.model = get_shared_evaluator_model()
    return evaluator


@st.cache_data
def load_reference_answers():
    import json
    queries_path = PROJECT_ROOT / "evaluation" / "generation_queries.json"
    ref_map = {}
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for q in data:
                ref_map[q["query"].strip().lower()] = q["expected_answer"]
    return ref_map

@st.cache_data
def load_benchmark_summary():
    import json
    summary_path = PROJECT_ROOT / "evaluation" / "response_quality" / "response_quality_results.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None



# Sidebar
with st.sidebar:
    st.markdown("### 💬 Chat")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    if st.session_state.messages:
        user_queries = [m for m in st.session_state.messages if m.get("role") == "user"]
        st.caption(f"Active Conversation: **{len(user_queries)}** {'turn' if len(user_queries) == 1 else 'turns'}")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    st.divider()
    st.header("⚙️ System Status")
    st.success("RAG Backend FROZEN & VALIDATED")
    
    st.subheader("Evaluation Metrics")
    st.markdown("""
    **Tests:** 46/46 PASSED  
    **Queries:** 82 Benchmark Queries  
    
    **Retrieval Performance:**
    - Hit@1: 97.56%
    - Hit@3: 100.00%
    - Hit@5: 100.00%
    - MRR: 0.9858
    
    **Response Quality Criteria:**
    - Correctness: Grounded strictly in context
    - Relevancy: Addresses the exact query intent
    - Faithfulness: No hallucinations
    """)
    st.divider()
    st.caption("Hybrid-RAG pipeline using Canonical RRF, MMR, and CrossEncoder multi-query aggregation.")

# Pipeline Analysis Function
def render_pipeline_analysis(res):
    st.markdown("### 🔬 RAG Pipeline Analysis")
    tabs = st.tabs(["🔄 Pipeline Trace", "📑 Context & Sources", "🔍 Retrieval Comparison", "📊 Response Quality"])
    
    with tabs[0]:
        st.subheader("Pipeline Trace")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1. Query Reformulation")
            for q in res.get("query_variants", []):
                st.info(q)
        with col2:
            st.markdown("#### 2. Fusion & Pool")
            st.metric("RRF Candidate Pool Size", len(res.get("fused_results", [])))
            st.caption(f"Configured FUSION_TOP_K: {FUSION_TOP_K}")
            
            # Show top items in fusion pool
            if res.get("fused_results"):
                st.markdown("**Top 3 Fused Results:**")
                for f in res.get("fused_results")[:3]:
                    st.caption(f"- {f.get('chunk_id')} (RRF: {f.get('rrf_score', 0):.4f})")
                    
        with col3:
            st.markdown("#### 3. CrossEncoder Reranking")
            st.metric("Final Top-K Context", len(res.get("final_results", [])))
            st.caption(f"Configured FINAL_TOP_K: {FINAL_TOP_K}")
            
    with tabs[1]:
        st.subheader("Context Supplied to LLM")
        st.text_area("Prompt Context", res.get("context", "No context available."), height=350)
        
    with tabs[2]:
        st.subheader("🔍 Retrieval Strategy & Hybrid RAG Comparison")
        st.caption("Compare the individual retrieval strategies with the complete Hybrid-RAG pipeline.")
        
        strat_tabs = st.tabs(["Naive", "Sentence Window", "Parent-Child", "BM25", "🔀 Hybrid RAG"])
        
        def render_strat(results):
            if not results:
                st.warning("No results for this strategy.")
                return
            df = pd.DataFrame([{
                "Query Variant": r.get('query_variant', ''),
                "Document": r.get('document_title', 'Unknown'),
                "Chunk ID": r.get('chunk_id', ''),
                "Score": round(r.get('retrieval_score', 0), 4)
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with strat_tabs[0]:
            st.caption("**Naive**: Dense retrieval over naive chunks.")
            render_strat(res.get("naive_results", []))
            
        with strat_tabs[1]:
            st.caption("**Sentence Window**: Dense retrieval over sentence-window representations.")
            render_strat(res.get("sw_results", []))
            
        with strat_tabs[2]:
            st.caption("**Parent-Child**: Dense retrieval over child chunks with parent context.")
            render_strat(res.get("pc_results", []))
            
        with strat_tabs[3]:
            st.caption("**BM25**: Lexical retrieval.")
            render_strat(res.get("bm25_results", []))
            
        with strat_tabs[4]:
            st.caption("**Hybrid RAG**: Final production pipeline combining dense and lexical retrieval, RRF fusion, and CrossEncoder reranking.")
            
            st.markdown("""
            **QUERY**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **MULTI-QUERY RETRIEVAL**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **DENSE + BM25**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **RRF**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **TOP-15**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **CROSSENCODER**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **FINAL TOP-5**  
            &nbsp;&nbsp;&nbsp;&nbsp;↓  
            **QWEN3-0.6B**
            """)
            
            st.markdown("### Final Hybrid RAG Results")
            st.caption("CrossEncoder Score: Query–passage relevance score used for reranking.")
            
            hybrid_df = []
            for i, r in enumerate(res.get("final_results", [])):
                hybrid_df.append({
                    "Rank": i + 1,
                    "Document": r.get('document_title', 'Unknown'),
                    "Section": r.get('section', 'N/A'),
                    "CrossEncoder Score": round(r.get('rerank_score', 0), 4),
                })
                
            st.dataframe(pd.DataFrame(hybrid_df), use_container_width=True, hide_index=True)
            
            st.markdown("### 📚 Final Top-5 Context")
            for i, r in enumerate(res.get("final_results", [])[:5]):
                with st.expander(f"Source {i+1}\n\n{r.get('document_title', 'Unknown')} - {r.get('section', 'N/A')}"):
                    st.markdown(r.get('text', 'No text available.'))

    with tabs[3]:
        st.subheader("Automated Response Quality Evaluation")
        st.caption("Calibrated semantic similarity, claim-level grounding verification, and multi-strategy comparison.")
        st.write("")

        eval_res = res.get("evaluation")
        if eval_res:
            c_info = eval_res.get("correctness", {})
            r_info = eval_res.get("relevancy", {})
            f_info = eval_res.get("faithfulness", {})

            c_score = c_info.get("score")
            ref_avail = c_info.get("reference_available", False)
            r_score = r_info.get("score", 0.0)
            f_score = f_info.get("score", 0.0)
            supp = f_info.get("supported_claims", 0)
            total_claims = f_info.get("total_claims", 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                if ref_avail:
                    st.info(f"**Correctness**\n\n### {c_score} / 5\nGround-Truth Match")
                else:
                    st.info(f"**Correctness**\n\n### N/A\nReference-Free Query")
            with col2:
                st.info(f"**Relevancy**\n\n### {r_score:.2f}\nQuery–Answer Alignment")
            with col3:
                pct = int(f_score * 100) if total_claims > 0 else 0
                st.info(f"**Faithfulness**\n\n### {f_score:.2f}\n{supp}/{total_claims} claims grounded ({pct}%)")

            st.write("")
            st.markdown("""
**Correctness (1–5):** Semantic and key-fact alignment against benchmark reference answer.  
**Relevancy (0–1):** Direct semantic relevance of generated answer to the user query.  
**Faithfulness (0–1):** Proportion of factual claims substantiated by retrieved context passages.
""")

            ref_match = res.get("reference_match")
            if ref_match:
                with st.expander(f"📖 Matched Benchmark Reference ({ref_match.get('match_similarity', 0):.0%} match)"):
                    st.markdown(f"**Benchmark Query:** {ref_match.get('query')}")
                    st.markdown(f"**Expected Answer:** {ref_match.get('expected_answer')}")
            elif not ref_avail:
                st.caption("ℹ️ *Note: Correctness is N/A because this is a custom query without a predefined ground-truth reference answer.*")
        else:
            st.warning("Response-quality evaluation is not available for this query.")

        st.divider()
        st.subheader("🔬 4-Strategy Response Quality Comparison")
        st.caption("Compare how **Naive RAG**, **Sentence Window**, **Parent-Child**, and **Hybrid RAG** perform on this prompt.")

        multi_eval = res.get("multi_strategy_eval")
        query_text = res.get("query", "")

        btn_key = f"btn_compare_{abs(hash(query_text))}"
        if not multi_eval:
            if st.button("⚡ Evaluate All 4 Retrieval Strategies", key=btn_key, type="secondary"):
                with st.spinner("Synthesizing answers and evaluating Naive, Sentence Window, and Parent-Child (reusing Hybrid RAG metrics)..."):
                    from src.evaluation.simple_evaluator import evaluate_all_rag_strategies
                    evaluator = load_evaluator()
                    ref_m = res.get("reference_match")
                    ref_text = ref_m.get("expected_answer") if ref_m else None
                    k_facts = ref_m.get("key_facts") if ref_m else None
                    multi_eval = evaluate_all_rag_strategies(
                        rag=load_rag(),
                        query=query_text,
                        rag_result=res,
                        evaluator=evaluator,
                        reference_answer=ref_text,
                        key_facts=k_facts,
                    )
                    res["multi_strategy_eval"] = multi_eval
                    st.rerun()

        if multi_eval:
            st.markdown("#### Performance Metrics by Strategy")
            summary_df = pd.DataFrame(multi_eval.get("summary_table", []))
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("#### Generated Responses by Strategy")
            strategies_data = multi_eval.get("strategies", {})
            for strat_name, strat_item in strategies_data.items():
                with st.expander(f"📝 {strat_name} Response"):
                    st.markdown(f"**Answer:**\n\n{strat_item.get('answer', 'N/A')}")
                    e = strat_item.get("evaluation", {})
                    c_val = f"{e.get('correctness', {}).get('score')}/5" if e.get('correctness', {}).get('reference_available') else "N/A"
                    r_val = f"{e.get('relevancy', {}).get('score', 0):.2f}"
                    f_val = f"{e.get('faithfulness', {}).get('score', 0):.2f}"
                    st.caption(f"Correctness: {c_val} | Relevancy: {r_val} | Faithfulness: {f_val}")

        # Benchmark reference section
        bench_summary = load_benchmark_summary()
        if bench_summary and "summary" in bench_summary:
            with st.expander("📈 Overall Benchmark Evaluation Summary (Standard Test Suite)"):
                b_rows = []
                for m_key, m_stats in bench_summary["summary"].items():
                    b_rows.append({
                        "Retrieval Strategy": m_key,
                        "Correctness (Mean ± Std)": f"{m_stats['correctness']['mean']:.2f} ± {m_stats['correctness']['std']:.2f}",
                        "Relevancy (Mean ± Std)": f"{m_stats['relevancy']['mean']:.2f} ± {m_stats['relevancy']['std']:.2f}",
                        "Faithfulness (Mean ± Std)": f"{m_stats['faithfulness']['mean']:.2f} ± {m_stats['faithfulness']['std']:.2f}",
                        "Benchmark Samples": m_stats.get("num_samples", 0),
                    })
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
rag = load_rag()

# ── Step 1: Input Capture & Prompt Determination ──
import streamlit.components.v1 as components

# Callback for quick-prompt suggestion cards (executes before script rerun)
def select_quick_prompt(q_text: str):
    st.session_state.pending_prompt = q_text

chat_val = st.chat_input("Ask a question about government schemes...")

# Determine active prompt: either from bottom chat input or clicked suggestion card
prompt = chat_val or st.session_state.pop("pending_prompt", None)

# Only show the welcome hero & suggestion cards when NO conversation has started AND NO query is in-flight
show_welcome = (len(st.session_state.messages) == 0 and not prompt)

if show_welcome:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 24px;">
        <div style="font-size: 3rem; margin-bottom: 8px;">🤖</div>
        <h1 style="font-size: 2.2rem; font-weight: 700; margin: 0; color: #F8FAFC; letter-spacing: -0.02em;">HYBRID-RAG</h1>
        <p style="font-size: 1.05rem; color: #94A3B8; margin-top: 6px; margin-bottom: 20px;">Government Welfare Scheme Research & Intelligence Assistant</p>
        <p style="font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px;">Suggested Questions</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.button(
            "🏠 **PMAY-U Eligibility**\n\nWhat are all the eligibility conditions for PMAY-U?",
            key="sugg_pmayu",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("What are all the eligibility conditions for PMAY-U?",)
        )
        st.button(
            "🌾 **PM-KISAN Financial Support**\n\nWhat is the annual financial benefit provided under PM-KISAN and how is it paid?",
            key="sugg_pmkisan",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("What is the annual financial benefit provided under PM-KISAN and how is it paid?",)
        )
        st.button(
            "🛡️ **Atal Pension Yojana (APY)**\n\nWhat is the age limit to join Atal Pension Yojana?",
            key="sugg_apy",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("What is the age limit to join Atal Pension Yojana?",)
        )
    with col2:
        st.button(
            "🏘️ **PMAY-G Rural Benefits**\n\nWhat are the benefits provided under PMAY-G?",
            key="sugg_pmayg",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("What are the benefits provided under PMAY-G?",)
        )
        st.button(
            "🛍️ **PM-SVANidhi Credit Facility**\n\nWhat are the benefits of PM-SVANidhi?",
            key="sugg_svanidhi",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("What are the benefits of PM-SVANidhi?",)
        )
        st.button(
            "⚖️ **PMAY-U vs PMAY-G Comparison**\n\nCompare PMAY-U and PMAY-G in terms of beneficiaries and housing support.",
            key="sugg_compare",
            use_container_width=True,
            on_click=select_quick_prompt,
            args=("Compare PMAY-U and PMAY-G in terms of beneficiaries and housing support.",)
        )

# ── Step 2: Client-side lock/unlock watcher (always active) ──
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;
    if (!parentDoc) return;
    const win = window.parent;

    function getChatElements() {
        const chatInput = parentDoc.querySelector('[data-testid="stChatInput"]');
        if (!chatInput) return {};
        const textarea = chatInput.querySelector('textarea');
        const submitBtn = chatInput.querySelector('[data-testid="stChatInputSubmitButton"]') || chatInput.querySelector('button');
        return { chatInput, textarea, submitBtn };
    }

    function isStreamlitRunning() {
        return Boolean(
            parentDoc.querySelector('[data-testid="stStatusWidgetRunningIcon"]') ||
            parentDoc.querySelector('[data-testid="stStatusWidgetRunningManIcon"]') ||
            parentDoc.querySelector('[data-testid="stAppRunningIcon"]') ||
            parentDoc.querySelector('[aria-label="Running..."]')
        );
    }

    function lockInput() {
        const { chatInput, textarea, submitBtn } = getChatElements();
        if (!chatInput) return;
        chatInput.classList.add("is-processing");
        if (submitBtn) {
            submitBtn.setAttribute("disabled", "true");
            submitBtn.setAttribute("title", "Already a query is running. Please wait...");
            submitBtn.style.pointerEvents = "none";
            submitBtn.style.opacity = "0.3";
            submitBtn.style.cursor = "not-allowed";
        }
        if (textarea) {
            textarea.setAttribute("title", "Already a query is running. Please wait...");
            textarea.style.cursor = "not-allowed";
        }
    }

    function unlockInput() {
        const { chatInput, textarea, submitBtn } = getChatElements();
        if (!chatInput) return;
        chatInput.classList.remove("is-processing");
        if (submitBtn) {
            submitBtn.removeAttribute("disabled");
            submitBtn.removeAttribute("title");
            submitBtn.style.pointerEvents = "";
            submitBtn.style.opacity = "";
            submitBtn.style.filter = "";
            submitBtn.style.cursor = "";
        }
        if (textarea) {
            textarea.removeAttribute("title");
            textarea.removeAttribute("disabled");
            textarea.style.cursor = "";
        }
    }

    function syncState() {
        const running = isStreamlitRunning();
        const timeSinceSend = Date.now() - (win.__lastRagSendTime || 0);
        if (!running) {
            if (timeSinceSend > 800 || !win.__lastRagSendTime) {
                unlockInput();
            }
        } else {
            lockInput();
        }
    }

    function bindListeners() {
        const { submitBtn, textarea } = getChatElements();
        function onSend() {
            win.__lastRagSendTime = Date.now();
            lockInput();
        }
        if (submitBtn && !submitBtn.dataset.lockBound) {
            submitBtn.dataset.lockBound = "true";
            submitBtn.addEventListener("click", onSend, true);
        }
        if (textarea && !textarea.dataset.lockBound) {
            textarea.dataset.lockBound = "true";
            textarea.addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey && textarea.value.trim().length > 0) {
                    onSend();
                }
            }, true);
        }
    }

    win.__unlockRagInput = unlockInput;
    bindListeners();
    syncState();

    if (!win.__ragStateWatcherActive) {
        win.__ragStateWatcherActive = true;
        const observer = new MutationObserver(() => { bindListeners(); syncState(); });
        observer.observe(parentDoc.body, { childList: true, subtree: true });
        setInterval(syncState, 200);
    }
})();
</script>
""", height=0, width=0)

# Helper function to render assistant turn
def render_assistant_turn(msg):
    st.markdown('<div class="ai-answer-header"><span class="ai-badge-icon">🤖</span><span class="ai-answer-title">AI Answer</span></div>', unsafe_allow_html=True)

    if msg.get("thought_steps"):
        t_time = msg.get("thought_time", 2.0)
        with st.status(f"Thought for {t_time}s • {len(msg['thought_steps'])} steps completed", expanded=False, state="complete"):
            for step in msg["thought_steps"]:
                st.markdown(f'<div class="thought-step-item">✓ {step}</div>', unsafe_allow_html=True)

    ans = msg.get("answer_text") or msg["content"].replace("### 🤖 AI Answer\n\n", "").replace("### 🤖 AI Answer", "")
    st.markdown(ans)

    if "rag_result" in msg:
        res = msg["rag_result"]
        if "sources" in res and res["sources"]:
            st.markdown("**Supporting Sources:**")
            for s in res["sources"]:
                st.markdown(f"- **{s.get('document_title', 'Unknown')}** (Section: *{s.get('section', 'N/A')}*, Chunk: `{s.get('chunk_id')}`)")

        with st.expander("🔬 View RAG Pipeline Analysis", expanded=False):
            render_pipeline_analysis(res)

# ── Step 3: Render full chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            render_assistant_turn(msg)

# ── Step 4: Process new query if one was submitted ──
if prompt:
    # a) Add user message to state and render it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message-text">{prompt}</div>', unsafe_allow_html=True)

    # b) Process and stream assistant response
    with st.chat_message("assistant"):
        st.markdown('<div class="ai-answer-header"><span class="ai-badge-icon">🤖</span><span class="ai-answer-title">AI Answer</span></div>', unsafe_allow_html=True)

        thought_steps = []
        start_time = time.time()

        with st.status("Thinking... Initializing pipeline", expanded=True) as status_box:
            def on_status_update(stage: str, msg_text: str, data: dict = None):
                if stage.endswith("_start"):
                    status_box.update(label=f"Thinking... {msg_text}", state="running")
                elif stage.endswith("_done"):
                    thought_steps.append(msg_text)
                    st.markdown(f'<div class="thought-step-item">✓ {msg_text}</div>', unsafe_allow_html=True)

            on_status_update("init_start", "Initializing Hybrid-RAG pipeline...")
            import importlib
            import src.generation.llm_generator as llm_gen
            import src.generation.context_builder as ctx_builder
            importlib.reload(llm_gen)
            importlib.reload(ctx_builder)

            from src.generation.context_builder import (
                filter_candidates_by_query_scope,
                build_context,
                extract_source_metadata,
            )

            inner_gen = rag.get_generator()
            if inner_gen is not None:
                inner_gen.__class__ = llm_gen.QwenGenerator
                llm_gen._GLOBAL_GENERATOR = inner_gen

            class _ScopedGenerator:
                """
                Wraps the cached QwenGenerator and applies query-scope filtering
                before every generate() call.

                This is the permanent fix for cross-scheme contamination:
                - Works regardless of @st.cache_resource on HybridRAG
                - Works regardless of module reload state on the generator instance
                - Intercepts candidates at the last moment before context is assembled
                - Rebuilds a clean, scheme-filtered context and returns filtered sources
                """
                def __init__(self, inner):
                    self._inner = inner

                def generate(self, query, context=None, candidates=None, top_k=5, filter_unrelated=True):
                    filtered_sources = []
                    if candidates and query and filter_unrelated:
                        # Prune candidates from schemes not mentioned in the query (Hybrid RAG)
                        scoped = filter_candidates_by_query_scope(candidates, query)
                        # Rebuild context from the filtered, scheme-relevant candidates only
                        context, _ = build_context(scoped, top_k=top_k, query=query, filter_unrelated=True)
                        # Build sources metadata from filtered candidates
                        filtered_sources = [
                            extract_source_metadata(i + 1, c)
                            for i, c in enumerate(scoped[:top_k])
                        ]
                    elif candidates and not filter_unrelated:
                        # Baseline RAGs: Pass raw chunks as-is without filtering unrelated schemes
                        context, _ = build_context(candidates, top_k=top_k, query=None, filter_unrelated=False)
                        filtered_sources = [
                            extract_source_metadata(i + 1, c)
                            for i, c in enumerate(candidates[:top_k])
                        ]
                    # Forward to the real (cached) model with appropriate context;
                    # pass candidates=None so the inner generator uses our context string.
                    gen_result = self._inner.generate(
                        query=query,
                        context=context,
                        candidates=None,
                        top_k=top_k,
                        filter_unrelated=filter_unrelated,
                    )
                    # Attach filtered sources if the inner generator didn't produce any
                    if filtered_sources and not gen_result.get("sources"):
                        gen_result["sources"] = filtered_sources
                    if gen_result.get("answer"):
                        gen_result["answer"] = llm_gen.clean_non_thinking_output(gen_result["answer"])
                    return gen_result

            scoped_generator = _ScopedGenerator(inner_gen)

            try:
                result = rag.answer(prompt, generator=scoped_generator, status_callback=on_status_update)
            except TypeError:
                result = rag.answer(prompt, generator=scoped_generator)

            on_status_update("evaluation_start", "Evaluating response quality metrics...")
            evaluator = load_evaluator()

            # Robust semantic matching against benchmark dataset with fallback
            ref_match = None
            if hasattr(evaluator, "find_reference_match"):
                try:
                    ref_match = evaluator.find_reference_match(prompt)
                except Exception:
                    ref_match = None

            if not ref_match:
                ref_map = load_reference_answers()
                direct_ans = ref_map.get(prompt.strip().lower())
                if direct_ans:
                    ref_match = {"expected_answer": direct_ans}

            ref_ans = ref_match.get("expected_answer") if ref_match else None
            key_facts = ref_match.get("key_facts") if ref_match else None


            # Retrieve the true grounded context from the pipeline
            context_str = result.get("context", "")
            if not context_str and result.get("final_results"):
                context_str = "\n\n".join(
                    [c.get("text", "") for c in result.get("final_results", []) if isinstance(c, dict)]
                )

            eval_res = evaluator.evaluate(
                query=prompt,
                generated_answer=result.get("answer", ""),
                reference_answer=ref_ans,
                retrieved_context=context_str,
                key_facts=key_facts,
            )
            result["evaluation"] = eval_res
            result["reference_match"] = ref_match

            r_score = eval_res.get("relevancy", {}).get("score", 0.0)
            f_score = eval_res.get("faithfulness", {}).get("score", 0.0)
            on_status_update("evaluation_done", f"Evaluated response quality: Relevancy {r_score:.2f}, Faithfulness {f_score:.2f}")


            elapsed = time.time() - start_time
            status_box.update(
                label=f"Thought for {elapsed:.1f}s • {len(thought_steps)} steps completed",
                state="complete",
                expanded=False
            )

        # Display answer
        st.markdown(result["answer"])

        # Display sources
        if "sources" in result and result["sources"]:
            st.markdown("**Supporting Sources:**")
            for s in result["sources"]:
                st.markdown(f"- **{s.get('document_title', 'Unknown')}** (Section: *{s.get('section', 'N/A')}*, Chunk: `{s.get('chunk_id')}`)")

        # Pipeline analysis
        with st.expander("🔬 View RAG Pipeline Analysis", expanded=False):
            render_pipeline_analysis(result)

        # c) Save completed response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "answer_text": result["answer"],
            "thought_steps": thought_steps,
            "thought_time": round(elapsed, 1),
            "rag_result": result,
        })

# ── Step 5: Post-render completion & smooth scroll ──
components.html("""
<script>
(function() {
    const win = window.parent;
    if (!win) return;
    win.__lastRagSendTime = 0;
    if (typeof win.__unlockRagInput === "function") {
        win.__unlockRagInput();
    }
    const doc = win.document;
    if (doc) {
        try {
            win.scrollTo({ top: doc.body.scrollHeight, behavior: 'smooth' });
        } catch(e) {}
    }
})();
</script>
""", height=0, width=0)



