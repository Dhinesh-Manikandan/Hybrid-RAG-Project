import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

class SimpleEvaluator:
    """
    Lightweight, automated evaluator using semantic similarity via SentenceTransformers.
    Computes Correctness (1-5), Relevancy (0-1), and Faithfulness (0-1) without LLM overhead.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        # Adjusted from 0.70 based on calibration. 
        # "maximum age is 60" gave 0.939 sim to "minimum age is 18"
        self.faithfulness_threshold = 0.95
        
    def _cosine_sim(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def evaluate(self, query: str, generated_answer: str, reference_answer: str, retrieved_context: str) -> dict:
        """
        Evaluate Correctness, Relevancy, and Faithfulness.
        """
        if not generated_answer or not isinstance(generated_answer, str) or not generated_answer.strip():
            return {
                "correctness": {"score": 1, "similarity": 0.0},
                "relevancy": {"score": 0.0},
                "faithfulness": {"score": 0.0, "supported_claims": 0, "total_claims": 0}
            }

        # 1. Correctness (1-5 based on cosine sim to reference)
        if not reference_answer or not isinstance(reference_answer, str) or not reference_answer.strip():
            c_score = 1
            c_sim = 0.0
        else:
            gen_emb = self.model.encode(generated_answer)
            ref_emb = self.model.encode(reference_answer)
            c_sim = float(self._cosine_sim(gen_emb, ref_emb))
            
            # Adjusted based on calibration: semantic paraphrases often drop to ~0.45 
            # with all-MiniLM-L6-v2 due to lexical differences.
            if c_sim >= 0.80: c_score = 5
            elif c_sim >= 0.60: c_score = 4
            elif c_sim >= 0.45: c_score = 3
            elif c_sim >= 0.30: c_score = 2
            else: c_score = 1
            
        # 2. Relevancy (0-1 based on cosine sim to query)
        if not query or not isinstance(query, str) or not query.strip():
            r_sim = 0.0
        else:
            # We prepend the query to the generated answer before embedding it.
            # This solves the asymmetric search anomaly where a direct, factual answer 
            # lacking the topic entity scores lower than a non-answering topical statement.
            q_emb = self.model.encode(query)
            contextualized_ans = query + " " + generated_answer
            ca_emb = self.model.encode(contextualized_ans)
            raw_r_sim = float(self._cosine_sim(q_emb, ca_emb))
            
            # A random unrelated sentence appended to the query typically yields ~0.75-0.80 sim 
            # purely because the query is duplicated. We normalize [0.75, 1.0] to [0.0, 1.0]
            # to ensure unrelated answers score low and relevant answers score high.
            r_sim = max(0.0, min(1.0, (raw_r_sim - 0.75) / (1.0 - 0.75)))
            
        # 3. Faithfulness (0-1 based on supported claims in context)
        # Split generated answer into sentences/claims (basic split on punctuation)
        claims = [c.strip() for c in re.split(r'[.!?\n]+', generated_answer) if len(c.strip()) >= 10]
        
        if not retrieved_context or not isinstance(retrieved_context, str) or not retrieved_context.strip():
            context_sentences = []
        else:
            context_sentences = [c.strip() for c in re.split(r'[.!?\n]+', retrieved_context) if len(c.strip()) >= 10]
            
        if not claims:
            f_score = 0.0
            supported = 0
            total = 0
        elif not context_sentences:
            f_score = 0.0
            supported = 0
            total = len(claims)
        else:
            claim_embs = self.model.encode(claims)
            context_embs = self.model.encode(context_sentences)
            
            supported = 0
            for c_emb in claim_embs:
                sims = [self._cosine_sim(c_emb, ctx_emb) for ctx_emb in context_embs]
                max_sim = max(sims) if sims else 0
                if max_sim >= self.faithfulness_threshold:
                    supported += 1
            
            total = len(claims)
            f_score = supported / total if total > 0 else 0.0
            
        return {
            "correctness": {"score": c_score, "similarity": c_sim},
            "relevancy": {"score": r_sim},
            "faithfulness": {"score": f_score, "supported_claims": supported, "total_claims": total}
        }
