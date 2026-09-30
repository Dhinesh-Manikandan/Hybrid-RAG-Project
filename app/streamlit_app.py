import sys
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

# Custom CSS for polished look
st.markdown(f"""
    <style>
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 4px; padding-top: 10px; padding-bottom: 10px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; }}
        h1, h2, h3 {{ color: #1E3A8A; }}
        
        /* Dynamic chat input placement */
        [data-testid="stChatInput"] > div {{
            width: 100% !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }}
        [data-testid="stChatInput"] textarea {{
            border-radius: 12px !important;
        }}

        /* Add bottom padding so content doesn't get hidden behind the pinned input */
        section[data-testid="stMain"] .block-container {{
            padding-bottom: 120px !important;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing frozen backend models & indexes (this happens only once)...")
def load_rag():
    rag = HybridRAG()
    # Force generator initialization to load the model into memory
    rag.get_generator()
    return rag

@st.cache_resource
def load_evaluator():
    from src.evaluation.simple_evaluator import SimpleEvaluator
    return SimpleEvaluator()

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


# Sidebar
with st.sidebar:
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
        st.caption("Scores are calculated using lightweight semantic similarity and claim-context matching.")
        st.write("")
        
        eval_res = res.get("evaluation")
        if eval_res:
            c_score = eval_res.get("correctness", {}).get("score", "N/A")
            r_score = eval_res.get("relevancy", {}).get("score", 0.0)
            f_score = eval_res.get("faithfulness", {}).get("score", 0.0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Correctness**\n\n{c_score} / 5")
            with col2:
                st.info(f"**Relevancy**\n\n{r_score:.2f}")
            with col3:
                st.info(f"**Faithfulness**\n\n{f_score:.2f}")
                
            st.write("")
            st.markdown("""
**Correctness:** Semantic similarity between generated answer and reference answer.  
**Relevancy:** Semantic similarity between the user query and generated answer.  
**Faithfulness:** Proportion of generated claims supported by the retrieved context.
""")
            
            if not eval_res.get("correctness", {}).get("similarity"):
                st.warning("Correctness score is 1 (minimum) because no reference answer exists in the evaluation dataset for this exact query.")
        else:
            st.warning("Response-quality evaluation is not available for this query.")

# Main UI
rag = load_rag()
quick_prompt = None

if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🤖<br>HYBRID-RAG</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>Government Scheme Assistant</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Ask questions about government schemes, eligibility, benefits, and application processes.</p>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.markdown("<h4 style='text-align: center;'>Try a question</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        q1 = st.button("PMAY-U Eligibility - What are all the eligibility conditions for PMAY-U?", use_container_width=True)
        q3 = st.button("PMAY-G Benefits - What are the benefits provided under PMAY-G?", use_container_width=True)
        q5 = st.button("APY Age Limit - What is the age limit to join Atal Pension Yojana?", use_container_width=True)
    
    with col2:
        q2 = st.button("PM-KISAN Benefits - What is the annual financial benefit provided under PM-KISAN and how is it paid?", use_container_width=True)
        q4 = st.button("PM-SVANidhi Benefits - What are the benefits of PM-SVANidhi?", use_container_width=True)
        q6 = st.button("PMAY-U vs PMAY-G - Compare PMAY-U and PMAY-G in terms of beneficiaries and housing support.", use_container_width=True)

    if q1: quick_prompt = "What are all the eligibility conditions for PMAY-U?"
    elif q2: quick_prompt = "What is the annual financial benefit provided under PM-KISAN and how is it paid?"
    elif q3: quick_prompt = "What are the benefits provided under PMAY-G?"
    elif q4: quick_prompt = "What are the benefits of PM-SVANidhi?"
    elif q5: quick_prompt = "What is the age limit to join Atal Pension Yojana?"
    elif q6: quick_prompt = "Compare PMAY-U and PMAY-G in terms of beneficiaries and housing support."

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "rag_result" in msg:
            res = msg["rag_result"]
            if "sources" in res and res["sources"]:
                st.markdown("**Supporting Sources:**")
                for s in res["sources"]:
                    st.markdown(f"- **{s.get('document_title', 'Unknown')}** (Section: {s.get('section', 'N/A')}, Chunk: `{s.get('chunk_id')}`)")
            
            with st.expander("🔬 View RAG Pipeline Analysis", expanded=False):
                render_pipeline_analysis(res)

# Capture chat input at the TOP LEVEL so Streamlit pins it to the bottom of the viewport.
chat_placeholder = st.empty()
chat_val = chat_placeholder.chat_input("Ask a question about government schemes...", key="enabled_chat")
prompt = chat_val if chat_val else quick_prompt

# Handle new user input
if prompt:
    # Explicitly disable the input box while processing this question
    chat_placeholder.chat_input("Processing your query...", disabled=True, key="disabled_chat")
    
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Processing Hybrid-RAG pipeline..."):
            result = rag.answer(prompt)
            
        with st.spinner("Evaluating response quality..."):
            evaluator = load_evaluator()
            ref_map = load_reference_answers()
            ref_ans = ref_map.get(prompt.strip().lower(), "")
            
            context_str = "\n\n".join([c.get('text', '') for c in result.get('sources', [])])
            
            eval_res = evaluator.evaluate(
                query=prompt,
                generated_answer=result.get("answer", ""),
                reference_answer=ref_ans,
                retrieved_context=context_str
            )
            result["evaluation"] = eval_res
            
        # Display AI Answer
        st.markdown("### 🤖 AI Answer")
        st.markdown(result["answer"])
        
        # Display Sources nicely BEFORE the expander
        if "sources" in result and result["sources"]:
            st.markdown("**Supporting Sources:**")
            for s in result["sources"]:
                st.markdown(f"- **{s.get('document_title', 'Unknown')}** (Section: {s.get('section', 'N/A')}, Chunk: `{s.get('chunk_id')}`)")
                
        # Render analysis expander AFTER sources
        with st.expander("🔬 View RAG Pipeline Analysis", expanded=False):
            render_pipeline_analysis(result)
            
        # Save assistant message to state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"### 🤖 AI Answer\n\n{result['answer']}",
            "rag_result": result
        })

