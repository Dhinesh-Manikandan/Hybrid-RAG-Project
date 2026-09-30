import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.simple_evaluator import SimpleEvaluator

def run_test():
    evaluator = SimpleEvaluator()
    
    test_cases = [
        {
            "query": "What is the age limit to join Atal Pension Yojana?",
            "A": "Subscribers must be between 18 and 40 years of age.",
            "B": "Subscribers must be at least 18 years of age.",
            "C": "Atal Pension Yojana is a pension scheme for unorganized sector workers.",
            "D": "PM-KISAN provides Rs. 6,000 per year to eligible farmers."
        },
        {
            "query": "What is the annual financial benefit provided under PM-KISAN and how is it paid?",
            "A": "Financial benefit of Rs. 6,000 per year payable in three equal installments of Rs. 2,000 each every four months.",
            "B": "Financial benefit of Rs. 6,000 per year is provided.",
            "C": "PM-KISAN is a central sector scheme to provide income support to landholding farmers.",
            "D": "Subscribers must be between 18 and 40 years of age."
        }
    ]
    
    for case in test_cases:
        query = case["query"]
        print(f"\nQuery: {query}")
        q_emb = evaluator.model.encode(query)
        
        for ans_type in ["A", "B", "C", "D"]:
            ans = case[ans_type]
            
            # Base sim
            a_emb = evaluator.model.encode(ans)
            base_sim = evaluator._cosine_sim(q_emb, a_emb)
            
            # Contextualized sim
            contextualized_ans = query + " " + ans
            ca_emb = evaluator.model.encode(contextualized_ans)
            ctx_sim = evaluator._cosine_sim(q_emb, ca_emb)
            
            # Normalized Context Sim (shift range so that unrelated is ~0)
            # The lowest possible contextualized sim for a random sentence appended to Q is around 0.70-0.80
            # Let's map [0.75, 1.0] -> [0.0, 1.0]
            norm_ctx = max(0, (ctx_sim - 0.75) / (1.0 - 0.75))
            
            print(f"[{ans_type}] Base: {base_sim:.4f} | Ctx: {ctx_sim:.4f} | NormCtx: {norm_ctx:.4f} | {ans}")

if __name__ == '__main__':
    run_test()
