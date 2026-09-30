import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.simple_evaluator import SimpleEvaluator

def run_test():
    evaluator = SimpleEvaluator()
    
    # Selected queries and hand-crafted answers for A, B, C, D
    test_cases = [
        {
            "query": "What is the age limit to join Atal Pension Yojana?",
            "A": "Subscribers must be between 18 and 40 years of age.",
            "B": "Subscribers must be at least 18 years of age.",
            "C": "Atal Pension Yojana is a pension scheme for unorganized sector workers.",
            "D": "PM-KISAN provides Rs. 6,000 per year to eligible farmers."
        },
        {
            "query": "What is the guaranteed monthly pension amount under Atal Pension Yojana upon turning 60?",
            "A": "Subscribers receive a guaranteed monthly pension of Rs. 1,000, Rs. 2,000, Rs. 3,000, Rs. 4,000, or Rs. 5,000 per month.",
            "B": "Subscribers receive a guaranteed monthly pension starting from Rs. 1,000.",
            "C": "The pension amount depends on the subscriber's contribution and age of joining.",
            "D": "The Centre provides 5 kg of free food grains per month to eligible beneficiaries."
        },
        {
            "query": "How much free food grain is supplied per month under PM-GKAY?",
            "A": "The Centre provides 5 kg of free food grains per month to eligible beneficiaries.",
            "B": "The Centre provides free food grains per month.",
            "C": "PM-GKAY was launched to help people during the COVID-19 pandemic.",
            "D": "A cashback of up to Rs. 100 per month is given to street vendors."
        },
        {
            "query": "What is the annual financial benefit provided under PM-KISAN and how is it paid?",
            "A": "Financial benefit of Rs. 6,000 per year payable in three equal installments of Rs. 2,000 each every four months.",
            "B": "Financial benefit of Rs. 6,000 per year is provided.",
            "C": "PM-KISAN is a central sector scheme to provide income support to landholding farmers.",
            "D": "Subscribers must be between 18 and 40 years of age."
        },
        {
            "query": "What is the initial working capital loan amount provided under PM SVANidhi?",
            "A": "An initial collateral-free working capital loan of Rs. 10,000 is provided, followed by Rs. 20,000 and Rs. 50,000 in subsequent tranches.",
            "B": "An initial collateral-free working capital loan of Rs. 10,000 is provided.",
            "C": "PM SVANidhi provides working capital loans to street vendors.",
            "D": "Financial benefit of Rs. 6,000 per year payable in three equal installments."
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
        print("-" * 50)
        
        for ans_type in ["A", "B", "C", "D"]:
            ans = case[ans_type]
            res = evaluator.evaluate(query, ans, ans, "dummy context")
            score = res["relevancy"]["score"]
            print(f"[{ans_type}] {score:.4f} | {ans}")

if __name__ == '__main__':
    run_test()
