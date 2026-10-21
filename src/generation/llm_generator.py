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
    "You are an expert government-scheme advisory assistant providing concise, accurate, and strictly grounded answers.\n\n"
    "Answer the user's question using ONLY the provided context.\n\n"
    "CRITICAL RULES:\n"
    "1. Concise & Direct: Answer ONLY what the user's question asks. Be concise, direct, and to the point. Do NOT include unnecessary background, unasked scheme objectives, or extra unrequested sections even if they are present in the context.\n"
    "2. Strict Grounding: Every statement must be directly supported by the provided context. Include only facts, figures, criteria, and details explicitly stated in the context. Never extrapolate, speculate, or invent facts.\n"
    "3. Strict Scope Discipline — No Cross-Scheme Contamination: Discuss ONLY the specific scheme(s) explicitly named in the user's question. If a passage mentions a different scheme, COMPLETELY IGNORE it.\n"
    "4. No Unrequested Comparisons or Dummy Rows: Do NOT include comparison tables, differences sections, or dummy placeholder rows for unmentioned schemes unless the user explicitly requested a comparison between multiple schemes.\n"
    "5. Clean Response: When answering a valid question, output ONLY the direct answer. Never append apologies, disclaimers, or out-of-scope warnings to an answer that has already addressed the user's question.\n"
    "6. Out-of-Scope Detection: ONLY if the user's question cannot be answered at all because it is completely unrelated to government welfare schemes or the context contains zero relevant information, respond ONLY with: "
    "'I\\'m sorry, I can only answer questions related to government welfare schemes covered in my knowledge base. "
    "Your question does not appear to be in scope. Please ask about a scheme such as PMAY, PM-KISAN, APY, PM SVANidhi, Stand-Up India, or PM Vishwakarma.'"
)

USER_PROMPT_TEMPLATE = (
    "Context:\n"
    "{context}\n\n"
    "Question:\n"
    "{query}\n\n"
    "Instructions: Answer the question comprehensively, accurately, and thoroughly using ONLY the relevant facts from the context above. Provide enough detail and complete information on all key criteria, numbers, and stipulations specifically asked. Focus strictly on the scheme asked in the question and ignore any unrelated schemes."
)


NO_CONTEXT_FALLBACK_ANSWER = "The information is not available in the retrieved sources."


def clean_non_thinking_output(text: str) -> str:
    """
    Remove any residual thinking/reasoning blocks to guarantee non-thinking output.
    Also strips accidental out-of-scope disclaimers or dummy comparison rows if the model
    already provided a substantive answer.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # If the response already contains a substantive answer, strip any trailing
    # out-of-scope disclaimer that the model may have erroneously appended.
    disclaimer_pattern = (
        r"\n*(?:I'?m\s+sorry,?\s+I\s+can\s+only\s+answer\s+questions\s+related\s+to\s+government\s+welfare\s+schemes[^\n]*"
        r"(?:knowledge\s+base[^\n]*)?(?:Please\s+ask\s+about\s+a\s+scheme[^\n]*)?)\.?\s*$"
    )
    stripped = re.sub(disclaimer_pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    if len(stripped) > 40:
        cleaned = stripped

    # Clean up any accidental dummy comparison rows like "| Other Schemes | Not specified | Not mentioned |"
    cleaned = re.sub(
        r"\n?\|?\s*Other Schemes\s*\|\s*Not (?:specified|mentioned)\s*\|\s*Not (?:specified|mentioned)\s*\|?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Clean up empty comparison table or dangling "### Key Differences" header if left empty or only dummy rows
    cleaned = re.sub(r"###\s*Key Differences\s*(\n\s*\|[^\n]+\|\s*)*$", "", cleaned, flags=re.DOTALL).strip()

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
            import os
            optimal_threads = min(8, os.cpu_count() or 4)
            try:
                torch.set_num_threads(optimal_threads)
            except Exception:
                pass
            self.model.to(self.device)

        self.model.eval()
        self._is_loaded = True
        print(f"LLM generator ready on device: {self.device}")

    def build_prompt(self, query: str, context: str) -> str:
        """
        Construct prompt using Qwen's official chat template in non-thinking mode.
        """
        system_prompt = SYSTEM_PROMPT
        
        # Multi-scheme comparison detection: only trigger if the user explicitly asks to compare
        # AND multiple schemes are mentioned in the query.
        compare_intent = any(kw in query.lower() for kw in [
            "compare", "comparison", "difference between", "differences between", 
            "versus", " vs "
        ])
        
        # Check how many distinct schemes are in the query
        try:
            from src.generation.context_builder import SCHEME_TAXONOMY
            schemes_in_query = set()
            q_lower = query.lower()
            for scheme_id, info in SCHEME_TAXONOMY.items():
                if any(alias in q_lower for alias in info["aliases"]):
                    schemes_in_query.add(scheme_id)
        except Exception:
            schemes_in_query = set()

        if compare_intent and len(schemes_in_query) >= 2:
            system_prompt += (
                "\n\nFor this multi-scheme comparison question, format your response as follows:\n"
                "1. Provide a concise section for each requested scheme with a markdown heading (e.g., `### Scheme Name`).\n"
                "2. Conclude with a '### Key Differences' section highlighting only the contrasts between the requested schemes.\n"
                "3. Do NOT include any unasked schemes or placeholders like 'Other Schemes'."
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
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT_TEMPLATE.format(query=query, context=context)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = GENERATION_CONTEXT_TOP_K,
        filter_unrelated: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate grounded answer from query and context/candidates.

        Args:
            query: User question string.
            context: Optional pre-built context string. If not provided, built from candidates.
            candidates: Optional list of retrieved and reranked candidate dicts.
            top_k: Number of sources to include if building context from candidates.
            filter_unrelated: If True (Hybrid RAG), applies query-scope filtering to prune
                             unrelated scheme chunks and score-margin filtering.
                             If False (baseline RAGs), preserves chunks 'as is'.

        Returns:
            Dict containing:
                "answer": Generated answer string.
                "sources": List of source attribution dictionaries.
        """
        # Determine context and sources
        sources: List[Dict[str, Any]] = []

        if candidates:
            from src.generation.context_builder import (
                build_context,
                filter_candidates_by_query_scope,
            )
            if filter_unrelated:
                # Hybrid RAG: Prune chunks from schemes not in query and apply score-margin filter
                scoped_candidates = filter_candidates_by_query_scope(candidates, query) if query else candidates
                built_context, built_sources = build_context(
                    scoped_candidates, top_k=top_k, query=query, filter_unrelated=True
                )
            else:
                # Baseline RAGs (Naive RAG, Sentence Window, Parent-Child):
                # Pass retrieved chunks directly 'as is' without filtering unrelated schemes.
                built_context, built_sources = build_context(
                    candidates, top_k=top_k, query=None, filter_unrelated=False
                )
            sources = built_sources
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
        eos_ids = [getattr(self.tokenizer, "eos_token_id", 151645)]
        im_end_id = getattr(self.tokenizer, "convert_tokens_to_ids", lambda x: None)("<|im_end|>")
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            "eos_token_id": eos_ids if len(eos_ids) > 1 else eos_ids[0],
            "repetition_penalty": 1.05,
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
    filter_unrelated: bool = True,
) -> Dict[str, Any]:
    """
    Public functional interface for grounded answer generation.

    Args:
        query: User question string.
        context: Optional pre-formatted context string.
        candidates: Optional retrieval candidates.
        top_k: Number of sources to include if candidates provided.
        generator: Optional custom or mocked generator instance.
        filter_unrelated: Whether to filter unrelated chunks (True for Hybrid RAG, False for baselines).

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    gen = generator if generator is not None else get_generator()
    return gen.generate(
        query=query,
        context=context,
        candidates=candidates,
        top_k=top_k,
        filter_unrelated=filter_unrelated,
    )
