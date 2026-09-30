# Response Quality Evaluation

## 1. Evaluation Objective
Evaluate the response quality of 4 RAG approaches using LLM-as-a-judge.

## 2. Dataset
Derived from `evaluation/generation_queries.json` with existing reference answers.

## 3. Number of Queries
1

## 4. RAG Strategies
Naive RAG, Sentence Window, Parent-Child, Hybrid RAG

## 5. Generator Model
Qwen/Qwen3-0.6B

## 6. Evaluator Model
Qwen/Qwen3-0.6B

## 7. Correctness Definition
Measures accuracy and relevance (1-5).

## 8. Relevancy Definition
Measures if response directly answers query and aligns with context (0-1).

## 9. Faithfulness Definition
Measures if response is supported by context (0-1).

## 10. LLM Evaluation Methodology
LLM-as-a-judge using specific prompts requesting structured JSON output.

## 11. Mean Calculation
Arithmetic mean of valid scores.

## 12. Standard Deviation Calculation
Sample standard deviation (`statistics.stdev`).

## 13. Results Table

TABLE I. RESPONSE QUALITY EVALUATION OF DIFFERENT RAG MODELS

| Model | Correctness (±std) | Relevancy (±std) | Faithfulness (±std) |
|---|---:|---:|---:|
| Naive RAG | 1.00 ± 0.00 | 0.53 ± 0.00 | 1.00 ± 0.00 |
| Sentence Window | 1.00 ± 0.00 | 0.53 ± 0.00 | 0.00 ± 0.00 |
| Parent-Child | 1.00 ± 0.00 | 0.53 ± 0.00 | 0.00 ± 0.00 |
| Hybrid RAG | 1.00 ± 0.00 | 0.55 ± 0.00 | 1.00 ± 0.00 |

## 14. Limitations
Evaluation relies on a smaller model (Qwen3-0.6B) as a judge.

## 15. Reproducibility Command
`python evaluation/response_quality/evaluate_response_quality.py --num-queries 30 --resume`
