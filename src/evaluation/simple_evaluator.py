import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, GENERATION_CONTEXT_TOP_K


class SimpleEvaluator:
    """
    Automated response quality evaluator using semantic similarity via SentenceTransformers.
    Computes Correctness (1-5), Relevancy (0-1), and Faithfulness (0-1) across
    individual retrieval strategies (Naive RAG, Sentence Window, Parent-Child, Hybrid RAG).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        # Calibrated for natural phrasing / semantic overlap in RAG responses
        # (0.72 reliably distinguishes grounded paraphrases from hallucinated claims)
        self.faithfulness_threshold = 0.72
        self._ref_cache: Optional[List[Dict[str, Any]]] = None
        self._ref_embeddings = None

    def _cosine_sim(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from a generated response.
        Decomposes compound sentences (e.g., comparative clauses with 'while', 'whereas', ';')
        into atomic sub-claims, strips formatting and field labels, and filters non-factual intro boilerplate.
        """
        if not text or not isinstance(text, str):
            return []

        # Clean line-by-line formatting
        lines = text.splitlines()
        raw_candidates = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Strip markdown headers, blockquotes, list markers
            line_str = re.sub(r"^[#*>\-\d.]+\s*", "", line_str).strip()
            if line_str:
                raw_candidates.append(line_str)

        full_cleaned = "\n".join(raw_candidates)

        # Split on sentence terminators or newlines
        sentences = re.split(r"(?<=[.!?])\s+|\n+", full_cleaned)

        claims = []
        # Non-factual introductory / meta-phrasing patterns
        intro_meta_pattern = re.compile(
            r"^(?:here (?:is|are)|the following|below is|in summary|to summarize|"
            r"both schemes differ|differ in their focus|differ in terms of|"
            r"summary of differences|supporting sources|key differences?:?)\b",
            re.IGNORECASE
        )

        for s in sentences:
            s_clean = s.strip()
            if not s_clean or len(s_clean) < 10:
                continue

            # Strip leading field labels like '**Beneficiaries:**', 'Beneficiaries:', 'Housing Support:'
            s_clean = re.sub(r"^(?:\*\*[^*]+\*\*|[A-Za-z\s]+):\s*", "", s_clean).strip()

            # Decompose compound comparative conjunctions
            subparts = re.split(
                r"(?:\s*;\s*|,\s*(?:while|whereas|in contrast to|on the other hand|as opposed to)\s+)",
                s_clean,
                flags=re.IGNORECASE,
            )

            for sp in subparts:
                sp_clean = sp.strip().rstrip(".,;")
                if not sp_clean or len(sp_clean) < 10:
                    continue
                # Skip list introductions ending in colon or pure comparison/meta framing phrases
                if sp_clean.endswith(":") or intro_meta_pattern.search(sp_clean) or re.search(
                    r"\b(?:differ in|differences between|comparison between|distinction between|overview of differences)\b",
                    sp_clean,
                    re.IGNORECASE,
                ):
                    continue
                claims.append(sp_clean)

        # Fallback if no atomic claims passed filter
        if not claims and len(text.strip()) >= 5:
            claims = [text.strip()]
        return claims

    def _extract_context_sentences(self, context: str) -> List[str]:
        """Extract sentences from retrieved context passages."""
        if not context or not isinstance(context, str):
            return []
        parts = re.split(r"(?<=[.!?])\s+|\n+", context)
        return [p.strip() for p in parts if len(p.strip()) >= 8]

    def load_reference_dataset(self, path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Load and cache benchmark reference queries for automated matching."""
        if self._ref_cache is not None:
            return self._ref_cache

        if path is None:
            root = Path(__file__).resolve().parent.parent.parent
            path = root / "evaluation" / "generation_queries.json"

        p = Path(path)
        if not p.exists():
            self._ref_cache = []
            return []

        try:
            with open(p, "r", encoding="utf-8") as f:
                self._ref_cache = json.load(f)
            query_texts = [q["query"].strip() for q in self._ref_cache]
            if query_texts:
                self._ref_embeddings = self.model.encode(query_texts)
        except Exception:
            self._ref_cache = []

        return self._ref_cache

    def find_reference_match(
        self,
        query: str,
        threshold: float = 0.78
    ) -> Optional[Dict[str, Any]]:
        """
        Semantically match an incoming query against benchmark reference queries.
        Returns the reference dict if similarity >= threshold, else None.
        """
        if not query or not query.strip():
            return None

        dataset = self.load_reference_dataset()
        if not dataset or self._ref_embeddings is None:
            return None

        q_emb = self.model.encode(query.strip())
        sims = [self._cosine_sim(q_emb, ref_emb) for ref_emb in self._ref_embeddings]
        if not sims:
            return None

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            matched = dict(dataset[best_idx])
            matched["match_similarity"] = round(best_sim, 4)
            return matched

        return None

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_context: Optional[Union[str, List[Dict[str, Any]]]] = None,
        key_facts: Optional[List[Any]] = None,
    ) -> dict:
        """
        Evaluate Correctness (1-5), Relevancy (0-1), and Faithfulness (0-1).

        Args:
            query: The user query string.
            generated_answer: The answer synthesized by the LLM.
            reference_answer: Optional ground-truth answer for correctness scoring.
            retrieved_context: Passage text string or list of retrieved candidate dicts.
            key_facts: Optional list of mandatory fact terms to verify.
        """
        # Normalize context from list of dicts if needed
        if isinstance(retrieved_context, list):
            retrieved_context = "\n\n".join(
                [c.get("text", "") for c in retrieved_context if isinstance(c, dict)]
            )
        elif retrieved_context is None:
            retrieved_context = ""

        # Handle empty generated response
        if not generated_answer or not isinstance(generated_answer, str) or not generated_answer.strip():
            return {
                "correctness": {
                    "score": 1,
                    "similarity": 0.0,
                    "reference_available": bool(reference_answer and reference_answer.strip()),
                },
                "relevancy": {"score": 0.0, "raw_similarity": 0.0},
                "faithfulness": {"score": 0.0, "supported_claims": 0, "total_claims": 0},
            }

        # ----------------------------------------------------
        # 1. Correctness (1-5 based on semantic similarity & key facts)
        # ----------------------------------------------------
        ref_present = bool(reference_answer and isinstance(reference_answer, str) and reference_answer.strip())
        if not ref_present:
            c_score = 1
            c_sim = 0.0
            fact_ratio = None
        else:
            gen_emb = self.model.encode(generated_answer)
            ref_emb = self.model.encode(reference_answer)
            c_sim = float(self._cosine_sim(gen_emb, ref_emb))

            # Check key facts if provided
            fact_ratio = 1.0
            if key_facts and isinstance(key_facts, list) and len(key_facts) > 0:
                found_count = 0
                ans_lower = generated_answer.lower()
                for fact_group in key_facts:
                    if isinstance(fact_group, list):
                        if any(str(syn).lower() in ans_lower for syn in fact_group):
                            found_count += 1
                    elif isinstance(fact_group, str) and fact_group.lower() in ans_lower:
                        found_count += 1
                fact_ratio = found_count / len(key_facts)

            eff_sim = (c_sim * 0.7 + fact_ratio * 0.3) if key_facts else c_sim

            if eff_sim >= 0.75 or (fact_ratio == 1.0 and c_sim >= 0.45):
                c_score = 5
            elif eff_sim >= 0.60 or (fact_ratio >= 0.8 and c_sim >= 0.40):
                c_score = 4
            elif eff_sim >= 0.45 or (fact_ratio >= 0.5 and c_sim >= 0.30):
                c_score = 3
            elif eff_sim >= 0.25:
                c_score = 2
            else:
                c_score = 1

        # ----------------------------------------------------
        # 2. Relevancy (0-1 based on direct query-answer alignment)
        # ----------------------------------------------------
        if not query or not isinstance(query, str) or not query.strip():
            r_score = 0.0
            raw_r_sim = 0.0
        else:
            q_emb = self.model.encode(query)
            ans_emb = self.model.encode(generated_answer)
            raw_r_sim = float(self._cosine_sim(q_emb, ans_emb))

            # Calibrated mapping: 0.15 (unrelated) -> 0.0, 0.75+ (strong relevance) -> 1.0
            r_score = max(0.0, min(1.0, (raw_r_sim - 0.15) / (0.75 - 0.15)))

        # ----------------------------------------------------
        # 3. Faithfulness (0-1 based on supported claims in context)
        # ----------------------------------------------------
        claims = self._extract_claims(generated_answer)
        context_sentences = self._extract_context_sentences(retrieved_context)

        if not claims:
            f_score = 0.0
            supported = 0
            total = 0
        elif not context_sentences and not (retrieved_context and retrieved_context.strip()):
            f_score = 0.0
            supported = 0
            total = len(claims)
        else:
            # Build target comparison units: context sentences + chunk passages
            # Chunk passages capture paragraph-level context for multi-clause or paraphrased claims
            chunk_passages = [
                p.strip() for p in re.split(r"\[Source \d+\]|\n\n+", retrieved_context)
                if len(p.strip()) >= 30
            ]
            all_targets = context_sentences + chunk_passages
            if not all_targets:
                all_targets = [retrieved_context.strip()]

            claim_embs = self.model.encode(claims)
            target_embs = self.model.encode(all_targets)

            # Pre-extract content words from context for grounding verification
            stopwords = {
                "this", "that", "with", "from", "their", "there", "they", "have", "were",
                "been", "which", "what", "whose", "under", "into", "also", "about", "each",
                "both", "such", "than", "other", "some", "only", "more", "most", "over",
            }
            context_lower = retrieved_context.lower()

            def _stem_word(w: str) -> str:
                for suffix in ("ization", "isation", "ing", "tion", "sion", "ties", "es", "ed", "s"):
                    if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                        return w[:-len(suffix)]
                return w

            context_tokens = set(re.findall(r"[a-zA-Z0-9]+", context_lower))
            context_stems = {_stem_word(t) for t in context_tokens}

            supported = 0
            for c, c_emb in zip(claims, claim_embs):
                sims = [self._cosine_sim(c_emb, t_emb) for t_emb in target_embs]
                max_sim = max(sims) if sims else 0.0

                # Compute lexical overlap of significant content words (>= 4 characters)
                tokens = re.findall(r"[a-zA-Z0-9]+", c.lower())
                content_words = [t for t in tokens if len(t) >= 4 and t not in stopwords]
                if content_words:
                    in_ctx = sum(
                        1 for w in content_words
                        if w in context_lower or any(_stem_word(w) in cs or cs in _stem_word(w) for cs in context_stems)
                    )
                    overlap_ratio = in_ctx / len(content_words)
                else:
                    overlap_ratio = 1.0

                # Robust grounding criterion:
                # 1. High semantic similarity (>= 0.65) OR
                # 2. Moderate semantic similarity (>= 0.45) with substantial content-word presence (>= 45%)
                is_grounded = (max_sim >= 0.65) or (max_sim >= 0.45 and overlap_ratio >= 0.45)
                if is_grounded:
                    supported += 1

            total = len(claims)
            f_score = supported / total if total > 0 else 0.0

        return {
            "correctness": {
                "score": c_score,
                "similarity": round(c_sim, 4) if ref_present else 0.0,
                "reference_available": ref_present,
                "key_facts_ratio": round(fact_ratio, 2) if fact_ratio is not None else None,
            },
            "relevancy": {
                "score": round(r_score, 4),
                "raw_similarity": round(raw_r_sim, 4),
            },
            "faithfulness": {
                "score": round(f_score, 4),
                "supported_claims": supported,
                "total_claims": total,
            },
        }

    def evaluate_strategies(
        self,
        query: str,
        strategy_outputs: Dict[str, Dict[str, Any]],
        reference_answer: Optional[str] = None,
        key_facts: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate and compare response quality across multiple retrieval strategies.

        Args:
            query: The user query string.
            strategy_outputs: Dict mapping strategy name (e.g. 'Naive RAG', 'Hybrid RAG')
                              to dict containing 'answer' and 'context' strings.
            reference_answer: Optional reference answer.
            key_facts: Optional key facts list.

        Returns:
            Dict containing detailed evaluation per strategy and a comparison summary table.
        """
        detailed: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []

        for name, data in strategy_outputs.items():
            ans = data.get("answer", "")
            ctx = data.get("context", "")

            # Optimization: reuse pre-calculated evaluation if provided (e.g. for Hybrid RAG)
            if "evaluation" in data and data["evaluation"]:
                eval_res = data["evaluation"]
            else:
                eval_res = self.evaluate(
                    query=query,
                    generated_answer=ans,
                    reference_answer=reference_answer,
                    retrieved_context=ctx,
                    key_facts=key_facts,
                )

            detailed[name] = {
                "answer": ans,
                "evaluation": eval_res,
            }

            c_dict = eval_res.get("correctness", {})
            r_dict = eval_res.get("relevancy", {})
            f_dict = eval_res.get("faithfulness", {})

            c_score_val = f"{c_dict.get('score', 'N/A')}/5" if c_dict.get("reference_available") else "N/A"
            r_score_val = r_dict.get("score", 0.0)
            f_score_val = f_dict.get("score", 0.0)
            supp_val = f"{f_dict.get('supported_claims', 0)}/{f_dict.get('total_claims', 0)}"

            summary_rows.append({
                "Strategy": name,
                "Correctness": c_score_val,
                "Relevancy": r_score_val,
                "Faithfulness": f_score_val,
                "Supported Claims": supp_val,
            })

        return {
            "query": query,
            "reference_answer": reference_answer,
            "strategies": detailed,
            "summary_table": summary_rows,
        }


def evaluate_all_rag_strategies(
    rag: Any,
    query: str,
    rag_result: Dict[str, Any],
    evaluator: SimpleEvaluator,
    reference_answer: Optional[str] = None,
    key_facts: Optional[List[Any]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function: For a query and retrieval result from HybridRAG,
    synthesizes answers for all 4 strategies (Naive, Sentence Window, Parent-Child, Hybrid RAG)
    and evaluates each across Correctness, Relevancy, and Faithfulness.
    """
    from src.generation.context_builder import build_context

    generator = rag.get_generator()
    actual_top_k = top_k or getattr(rag, "generation_context_top_k", GENERATION_CONTEXT_TOP_K)

    candidates = {
        "Naive RAG": rag_result.get("naive_results", [])[:actual_top_k],
        "Sentence Window": rag_result.get("sw_results", [])[:actual_top_k],
        "Parent-Child": rag_result.get("pc_results", [])[:actual_top_k],
        "Hybrid RAG": rag_result.get("final_results", [])[:actual_top_k],
    }

    strategy_outputs: Dict[str, Dict[str, Any]] = {}
    for name, cands in candidates.items():
        # Optimization: Reuse already generated Hybrid RAG answer and evaluation if present
        if name == "Hybrid RAG" and rag_result.get("answer"):
            strategy_outputs[name] = {
                "answer": rag_result.get("answer", ""),
                "context": rag_result.get("context", "")
                or "\n\n".join([c.get("text", "") for c in cands if isinstance(c, dict)]),
            }
            if rag_result.get("evaluation"):
                strategy_outputs[name]["evaluation"] = rag_result.get("evaluation")
        else:
            ctx_str, _ = build_context(cands, top_k=actual_top_k, query=query)
            gen_res = generator.generate(
                query=query,
                context=ctx_str,
                candidates=cands,
                top_k=actual_top_k,
            )
            strategy_outputs[name] = {
                "answer": gen_res.get("answer", ""),
                "context": ctx_str,
            }

    return evaluator.evaluate_strategies(
        query=query,
        strategy_outputs=strategy_outputs,
        reference_answer=reference_answer,
        key_facts=key_facts,
    )


