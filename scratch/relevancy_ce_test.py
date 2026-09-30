import sys
from pathlib import Path
from sentence_transformers import CrossEncoder

def run_test():
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')
    
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
        
        for ans_type in ["A", "B", "C", "D"]:
            ans = case[ans_type]
            score = model.predict([query, ans])
            print(f"[{ans_type}] CE Score: {score:.4f} | {ans}")

if __name__ == '__main__':
    run_test()
