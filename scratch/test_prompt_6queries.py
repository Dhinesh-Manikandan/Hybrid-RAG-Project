import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieve import HybridRAG
from src.evaluation.simple_evaluator import SimpleEvaluator

queries = [
    "What is the annual financial benefit provided under PM-KISAN and how is it paid?",
    "What are the eligibility conditions for PMAY-U?",
    "What are the benefits provided under PMAY-G?",
    "What are the benefits of PM SVANidhi?",
    "How can I apply for PMAY-U?",
    "What is the age limit to join Atal Pension Yojana?",
]

def main():
    print("Loading RAG...")
    rag = HybridRAG()
    rag.get_generator() # ensure loaded
    
    print("Loading Evaluator...")
    evaluator = SimpleEvaluator()

    for i, q in enumerate(queries):
        print(f"\n======================\nQUERY {i+1}: {q}")
        res = rag.answer(q)
        print("\nANSWER:")
        print(res["answer"])
        
        context_str = "\n\n".join([c.get('text', '') for c in res.get('sources', [])])
        eval_res = evaluator.evaluate(
            query=q,
            generated_answer=res["answer"],
            reference_answer="", # dummy
            retrieved_context=context_str
        )
        print(f"\nCorrectness: {eval_res['correctness']} | Relevancy: {eval_res['relevancy']} | Faithfulness: {eval_res['faithfulness']}")
        print(f"Sources Count: {len(res['sources'])}")

if __name__ == "__main__":
    main()
