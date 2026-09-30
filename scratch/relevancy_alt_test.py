import sys
import re
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.simple_evaluator import SimpleEvaluator

def get_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"what", "is", "the", "to", "and", "of", "in", "for", "how", "much", "many", "under", "a", "an"}
    return set([w for w in words if w not in stopwords])

def lexical_overlap(q, a):
    qw = get_words(q)
    aw = get_words(a)
    if not qw: return 0.0
    return len(qw.intersection(aw)) / len(qw)

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
        },
        {
            "query": "What cashback incentive is given for digital transactions under PM SVANidhi?",
            "A": "A cashback of up to Rs. 100 per month is given to street vendors for digital transactions.",
            "B": "A cashback is given to street vendors for digital transactions.",
            "C": "PM SVANidhi encourages street vendors to adopt digital transactions.",
            "D": "The Centre provides 5 kg of free food grains per month."
        }
    ]
    
    for case in test_cases:
        query = case["query"]
        print(f"\nQuery: {query}")
        
        q_emb = evaluator.model.encode(query)
        
        for ans_type in ["A", "B", "C", "D"]:
            ans = case[ans_type]
            a_emb = evaluator.model.encode(ans)
            
            sim = evaluator._cosine_sim(q_emb, a_emb)
            overlap = lexical_overlap(query, ans)
            
            # Combine them: maybe sim + overlap?
            combined = (sim * 0.5) + (overlap * 0.5)
            
            print(f"[{ans_type}] Sim: {sim:.4f} | Overlap: {overlap:.4f} | Comb: {combined:.4f} | {ans}")

if __name__ == '__main__':
    run_test()
