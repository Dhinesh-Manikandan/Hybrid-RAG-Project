import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.simple_evaluator import SimpleEvaluator

def run_calibration():
    evaluator = SimpleEvaluator()
    print("="*60)
    print("1. CORRECTNESS CALIBRATION")
    print("="*60)
    
    ref = "The minimum age to join Atal Pension Yojana is 18 years and the maximum age is 40 years."
    
    cases_c = {
        "CASE A (Exact Match)": ref,
        "CASE B (Clear Paraphrase)": "People can enroll in APY from the age of 18 up to the age of 40.",
        "CASE C (Partially Correct)": "People can join APY from 18 years of age.",
        "CASE D (Incorrect)": "People can join APY between 25 and 60 years.",
        "CASE E (Unrelated)": "PM-KISAN provides financial assistance to eligible farmers."
    }
    
    for name, gen in cases_c.items():
        res = evaluator.evaluate("query", gen, ref, "context")
        print(f"\n{name}")
        print(f"Generated: {gen}")
        print(f"Similarity: {res['correctness']['similarity']:.4f}")
        print(f"Score: {res['correctness']['score']} / 5")
        
    print("\n" + "="*60)
    print("2. RELEVANCY CALIBRATION")
    print("="*60)
    
    query = "What is the age limit to join Atal Pension Yojana?"
    
    cases_r = {
        "CASE A (Highly Relevant)": "The minimum age is 18 years and the maximum age is 40 years.",
        "CASE B (Related but incomplete)": "Atal Pension Yojana is a government pension scheme.",
        "CASE C (Unrelated)": "PM-KISAN provides Rs. 6,000 per year to eligible farmers."
    }
    
    for name, ans in cases_r.items():
        res = evaluator.evaluate(query, ans, "ref", "context")
        print(f"\n{name}")
        print(f"Answer: {ans}")
        print(f"Relevancy Score: {res['relevancy']['score']:.4f}")

    print("\n" + "="*60)
    print("3. FAITHFULNESS CALIBRATION")
    print("="*60)
    
    cases_f = [
        ("CASE A (Fully Supported)", 
         "The minimum age to join Atal Pension Yojana is 18 years and the maximum age is 40 years.",
         "The minimum age to join Atal Pension Yojana is 18 years and the maximum age is 40 years."),
        ("CASE B (Partially Supported)",
         "The minimum age to join Atal Pension Yojana is 18 years.",
         "The minimum age to join Atal Pension Yojana is 18 years and the maximum age is 40 years."),
        ("CASE C (Unsupported)",
         "The minimum age to join Atal Pension Yojana is 18 years.",
         "The maximum age to join Atal Pension Yojana is 60 years."),
        ("CASE D (Completely Unrelated)",
         "PM-KISAN provides financial assistance to eligible farmers.",
         "The minimum age to join Atal Pension Yojana is 18 years.")
    ]
    
    for name, context, ans in cases_f.items() if isinstance(cases_f, dict) else cases_f:
        res = evaluator.evaluate("query", ans, "ref", context)
        sim = 0
        if isinstance(res['faithfulness'], dict):
            # We want to print the similarity. We can just encode it here.
            ctx_embs = evaluator.model.encode([context])
            ans_embs = evaluator.model.encode([ans])
            sim = evaluator._cosine_sim(ctx_embs[0], ans_embs[0])
            
        print(f"\n{name}")
        print(f"Context: {context}")
        print(f"Answer: {ans}")
        print(f"Max Sim: {sim:.4f}")
        print(f"Faithfulness Score: {res['faithfulness']['score']:.4f}")
        print(f"Supported Claims: {res['faithfulness']['supported_claims']} / {res['faithfulness']['total_claims']}")

if __name__ == '__main__':
    run_calibration()
