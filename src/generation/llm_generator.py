"""
llm_generator.py -- Grounded answer generation using Qwen3-8B in non-thinking mode.

Adheres strictly to the frozen Hybrid-RAG retrieval architecture:
- Loads Qwen/Qwen3-8B via Hugging Face Transformers.
- Employs automatic device placement (CUDA where available, sensible CPU fallback).
- Configured explicitly for non-thinking mode (no chain-of-thought / hidden reasoning).
- Strict grounded RAG prompt preventing hallucinations or pretrained knowledge leaks.
- Structured response preserving source attribution metadata.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    GENERATION_CONTEXT_TOP_K,
    LLM_ENABLE_THINKING,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from src.generation.context_builder import build_context


# Grounded RAG prompt template
SYSTEM_PROMPT = (
    "You are a grounded government-scheme question answering assistant.\n\n"
    "Answer the user's question using ONLY the provided context.\n\n"
    "IMPORTANT RULES:\n"
    "1. Answer the question directly and naturally.\n"
    "2. Use all relevant information in the context, not just the first matching passage.\n"
    "3. For complex questions, provide a sufficiently detailed answer with an introduction and structured bullet points or sections.\n"
    "4. Do not refer to retrieved passages as 'Source 1', 'Source 2', 'first source', 'second source', etc.\n"
    "5. Do not mention chunks, embeddings, retrieval, RRF, CrossEncoder, or internal system details.\n"
    "6. Never invent information that is not present in the context.\n"
    "7. For a comparison question, check whether the context contains enough information about EVERY entity requested by the user.\n"
    "8. If information about one or more requested entities is missing, explicitly say that the available context is insufficient for that part of the comparison. Do not guess.\n"
    "9. Preserve all numbers, amounts, dates, conditions, and eligibility requirements exactly as supported by the context.\n"
    "10. For simple questions, answer concisely. For complex or comparison questions, provide a detailed structured answer.\n"
    "11. Do not repeat the same information.\n"
    "12. Do not add filler just to make the answer longer."
)

USER_PROMPT_TEMPLATE = (
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer the question using only the context above."
)

NO_CONTEXT_FALLBACK_ANSWER = "The information is not available in the retrieved sources."


def clean_non_thinking_output(text: str) -> str:
    """
    Remove any residual thinking/reasoning blocks to guarantee non-thinking output.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned


class QwenGenerator:
    """
    LLM generation manager for Qwen3-8B.
    Loads model and tokenizer once and reuses them for all queries.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        enable_thinking: bool = LLM_ENABLE_THINKING,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        # Allow dependency injection (for unit tests / mock models)
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._is_loaded = True
            return

        self.tokenizer = None
        self.model = None
        self.device = device
        self._is_loaded = False

    def load_model(self) -> None:
        """Load tokenizer and model with automatic device placement and dtype."""
        if self._is_loaded:
            return

        print(f"Loading LLM generator: {self.model_name}...")

        # Device determination
        has_cuda = torch.cuda.is_available()
        if self.device is None:
            self.device = "cuda" if has_cuda else "cpu"

        # Dtype selection
        if has_cuda:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load model with automatic device placement
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if has_cuda:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        if not has_cuda:
            self.model.to(self.device)

        self.model.eval()
        self._is_loaded = True
        print(f"LLM generator ready on device: {self.device}")

    def build_prompt(self, query: str, context: str) -> str:
        """
        Construct prompt using Qwen's official chat template in non-thinking mode.
        """
        system_prompt = SYSTEM_PROMPT
        
        # Complex question detection (Step 4)
        comparison_keywords = [
            "compare", "comparison", "differences", "difference between", 
            "versus", "benefits and eligibility", "advantages and disadvantages",
            "multiple schemes", "multiple entities", "and", "for each", 
            "respectively", "in terms of"
        ]
        
        q_lower = query.lower()
        if any(kw in q_lower for kw in comparison_keywords):
            system_prompt += (
                "\n\nFor a comparison question, provide:\n"
                "1. A short introductory statement.\n"
                "2. A separate section for each entity that is supported by the context.\n"
                "3. A final summary of the differences.\n"
                "4. If an entity is missing from the context, explicitly identify it as unavailable instead of guessing."
            )
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query, context=context)},
        ]

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Qwen chat template supports enable_thinking=False
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                return prompt_text
            except TypeError:
                # If apply_chat_template doesn't accept enable_thinking parameter
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt_text

        # Fallback raw prompt formatting
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT_TEMPLATE.format(query=query, context=context)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = GENERATION_CONTEXT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Generate grounded answer from query and context/candidates.

        Args:
            query: User question string.
            context: Optional pre-built context string. If not provided, built from candidates.
            candidates: Optional list of retrieved and reranked candidate dicts.
            top_k: Number of sources to include if building context from candidates.

        Returns:
            Dict containing:
                "answer": Generated answer string.
                "sources": List of source attribution dictionaries.
        """
        # Determine context and sources
        sources: List[Dict[str, Any]] = []

        if candidates:
            built_context, built_sources = build_context(candidates, top_k=top_k)
            sources = built_sources
            if context is None:
                context = built_context
        elif context is None:
            context = ""

        # Safe handling of empty context
        if not context or not context.strip():
            return {
                "answer": NO_CONTEXT_FALLBACK_ANSWER,
                "sources": [],
            }

        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # Build prompt
        prompt_text = self.build_prompt(query, context)

        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate tokens
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }

        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Slice off input tokens to keep only generated response
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Guarantee non-thinking mode (strip any thinking tags)
        answer = clean_non_thinking_output(raw_answer)

        return {
            "answer": answer,
            "sources": sources,
        }


# Singleton generator instance for reuse
_GLOBAL_GENERATOR: Optional[QwenGenerator] = None


def get_generator() -> QwenGenerator:
    """Retrieve or initialize the global QwenGenerator instance."""
    global _GLOBAL_GENERATOR
    if _GLOBAL_GENERATOR is None:
        _GLOBAL_GENERATOR = QwenGenerator()
    return _GLOBAL_GENERATOR


def generate_answer(
    query: str,
    context: Optional[str] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    top_k: int = GENERATION_CONTEXT_TOP_K,
    generator: Optional[QwenGenerator] = None,
) -> Dict[str, Any]:
    """
    Public functional interface for grounded answer generation.

    Args:
        query: User question string.
        context: Optional pre-formatted context string.
        candidates: Optional retrieval candidates.
        top_k: Number of sources to include if candidates provided.
        generator: Optional custom or mocked generator instance.

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    gen = generator if generator is not None else get_generator()
    return gen.generate(query=query, context=context, candidates=candidates, top_k=top_k)
