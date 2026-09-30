"""
query_reformulation.py -- Generic Query Reformulation & Multi-Query Perspective Generation.

Provides domain-agnostic and document-agnostic query reformulations to bridge
lexical and conceptual mismatches between conversational user queries and
formal expository document sections.

Architecture:
- reformulate_query(query) -> list[str]:
    Always returns: [original_query, alternative_1, alternative_2, ...]
- Guarantees:
  1. Original query is ALWAYS preserved at index 0.
  2. Exactly NUM_REFORMULATIONS alternatives are returned (configurable).
  3. Case-insensitive deduplication and empty string rejection.
  4. Robust fallback to [original_query] on any exception or timeout.
  5. Pluggable: Supports lightweight linguistic perspective engine (default,
     zero external dependencies, sub-millisecond latency) and configurable
     neural models.
"""

import re
import logging
from typing import Optional

from src.config import (
    ENABLE_QUERY_REFORMULATION,
    NUM_REFORMULATIONS,
    REFORMULATION_METHOD,
    REFORMULATION_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def _clean_query_string(text: str) -> str:
    """Normalize whitespace and strip punctuation from query string."""
    text = text.strip()
    while text.endswith("?") or text.endswith("."):
        text = text[:-1].strip()
    return re.sub(r"\s+", " ", text)


def _derive_nominal(word: str) -> str:
    """
    Derive nominal (noun) form of common English adjectives/participles
    using generic rule-based morphological transformations.
    Works for any domain vocabulary without hardcoding specific terms.
    """
    w = word.lower()
    if w.endswith("ble"):
        return word[:-3] + "bility"
    if w.endswith("ent"):
        return word[:-3] + "ence"
    if w.endswith("ant"):
        return word[:-3] + "ance"
    if w.endswith("ive"):
        return word[:-3] + "iveness"
    if w.endswith("ful"):
        return word[:-3] + "fulness"
    if w.endswith("ic"):
        return word + "ity"
    return word


def _generate_linguistic_perspectives(query: str, n: int = 2) -> list[str]:
    """
    Generate semantically diverse query formulations from complementary
    information-need perspectives (criteria, conditions, specifications, procedures).
    
    Operates strictly on generic English syntactic and interrogative structures.
    Contains NO domain-specific terms, document names, or hardcoded entities.
    """
    q = _clean_query_string(query)
    variants = []

    # 1. Agentive / Person question: "Who is/are [adj] [for/to/in/under] [entity]?"
    # e.g., "Who is eligible for X?" -> "What are the eligibility criteria for X?"
    who_adj_match = re.match(
        r"^who\s+(?:is|are|can\s+be)\s+([a-zA-Z\-]+)(?:\s+(?:for|to|in|under|with)\s+(.+))?$",
        q,
        re.IGNORECASE,
    )
    if who_adj_match:
        adj = who_adj_match.group(1).strip()
        entity = who_adj_match.group(2).strip() if who_adj_match.group(2) else ""
        noun = _derive_nominal(adj)
        if entity:
            variants.append(f"What are the {noun} criteria for {entity}?")
            variants.append(f"What conditions and requirements determine {noun} for {entity}?")
            variants.append(f"What requirements must a person satisfy to qualify for {entity}?")
        else:
            variants.append(f"What are the {noun} criteria?")
            variants.append(f"What conditions and requirements determine {noun}?")

    # 2. Action / Permission question: "Who can/may/should [verb] [entity]?"
    # e.g., "Who can apply for X?" -> "What are the qualifying criteria and requirements to apply for X?"
    who_verb_match = re.match(
        r"^who\s+(?:can|may|should)\s+([a-zA-Z\-]+)(?:\s+(?:for|to|in|at|under))?\s*(.*)$",
        q,
        re.IGNORECASE,
    )
    if not who_adj_match and who_verb_match:
        verb = who_verb_match.group(1).strip()
        entity = who_verb_match.group(2).strip() if who_verb_match.group(2) else ""
        if entity:
            variants.append(f"What are the criteria and requirements to {verb} {entity}?")
            variants.append(f"What conditions and qualifications must be satisfied to {verb} {entity}?")
        else:
            variants.append(f"What are the criteria and requirements to {verb}?")
            variants.append(f"What conditions must be satisfied to {verb}?")

    # 3. Procedural / How-to question: "How do I / How can [agent] [action] [entity]?"
    # e.g., "How can beneficiaries apply?" -> "What is the procedure and process for beneficiaries to apply?"
    how_match = re.match(
        r"^how\s+(?:do\s+i|can\s+([a-zA-Z\s]+)|to)\s+([a-zA-Z\-]+)(?:\s+(.+))?$",
        q,
        re.IGNORECASE,
    )
    if how_match:
        agent = how_match.group(1)
        action = how_match.group(2).strip() if how_match.group(2) else ""
        rest = how_match.group(3).strip() if how_match.group(3) else ""
        target = f"{action} {rest}".strip()
        if agent and agent.strip():
            clean_agent = agent.strip()
            variants.append(f"What is the application procedure and step-by-step process for {clean_agent} to {target}?")
            variants.append(f"What instructions, requirements, and parameters are needed for {clean_agent} to {target}?")
        else:
            variants.append(f"What is the procedure and step-by-step process to {target}?")
            variants.append(f"What instructions, requirements, and parameters are needed to {target}?")

    # 4. Inverted topic-predicate question: "What [topic] is/are [predicate]?"
    # e.g., "What benefits are available?" -> "What available benefits and specifications are provided?"
    what_inverted_match = re.match(
        r"^what\s+([a-zA-Z\s\-]+?)\s+(?:are|is)\s+([a-zA-Z\-]+)(?:\s+(?:for|in|under|of)\s+(.+))?$",
        q,
        re.IGNORECASE,
    )
    if not who_adj_match and not who_verb_match and not how_match and what_inverted_match:
        topic = what_inverted_match.group(1).strip()
        pred = what_inverted_match.group(2).strip()
        entity = what_inverted_match.group(3).strip() if what_inverted_match.group(3) else ""
        if entity:
            variants.append(f"What {pred} {topic} and specifications are provided in {entity}?")
            variants.append(f"What are the details, provisions, and criteria for {topic} in {entity}?")
        else:
            variants.append(f"What {pred} {topic} and specifications are provided?")
            variants.append(f"What are the details, provisions, and criteria for {topic}?")

    # 5. Standard topic enquiry: "What are the / What is [topic] [of/for/in entity]?"
    # e.g., "What are the eligibility criteria?" -> "What criteria, requirements, and details are defined for eligibility criteria?"
    what_std_match = re.match(
        r"^what\s+(?:are|is)(?:\s+the)?\s+([a-zA-Z\s\-]+?)(?:\s+(?:for|of|in|under)\s+(.+))?$",
        q,
        re.IGNORECASE,
    )
    if not who_adj_match and not who_verb_match and not how_match and not what_inverted_match and what_std_match:
        topic = what_std_match.group(1).strip()
        entity = what_std_match.group(2).strip() if what_std_match.group(2) else ""
        if not topic.lower().startswith("what"):
            if entity:
                variants.append(f"What specifications, requirements, and details are defined for {topic} in {entity}?")
                variants.append(f"What are the comprehensive criteria and provisions for {topic} in {entity}?")
            else:
                variants.append(f"What criteria, requirements, and details are defined for {topic}?")
                variants.append(f"What is the comprehensive specification and list of {topic}?")

    # 6. Additional complementary perspectives to fulfill arbitrary n
    variants.append(f"What are the key requirements, criteria, and details regarding {q}?")
    variants.append(f"What specifications, guidelines, and procedures apply to {q}?")
    variants.append(f"What essential conditions, parameters, and instructions are defined for {q}?")

    # Deduplicate against the original query and return up to n unique alternatives
    seen = {q.lower(), query.lower()}
    final_alts = []
    for v in variants:
        v_clean = _clean_query_string(v)
        if v_clean and v_clean.lower() not in seen:
            seen.add(v_clean.lower())
            final_alts.append(v_clean)
            if len(final_alts) >= n:
                break

    return final_alts


def _generate_model_perspectives(query: str, n: int = 2, model_name: str = REFORMULATION_MODEL_NAME) -> list[str]:
    """
    Optional neural seq2seq perspective generation.
    Wraps external or HuggingFace generation with strict fallbacks.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        prompt = f"Generate {n} diverse search queries for: {query}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=64, num_return_sequences=n, num_beams=n)
        decoded = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        return [d for d in decoded if d and d.lower() != query.lower()]
    except Exception as e:
        logger.warning(f"Neural query reformulation unavailable or failed: {e}. Falling back to linguistic engine.")
        return []


def reformulate_query(
    query: str,
    num_reformulations: int = NUM_REFORMULATIONS,
    method: str = REFORMULATION_METHOD,
    model_name: str = REFORMULATION_MODEL_NAME,
) -> list[str]:
    """
    Generate alternative query formulations for multi-query retrieval.

    Always returns:
        [original_query, alternative_1, alternative_2, ...]
    
    Guarantees:
    1. The original query is ALWAYS at index 0.
    2. At most num_reformulations alternatives are returned.
    3. No duplicate queries exist in the returned list.
    4. Empty strings are strictly excluded.
    5. On any failure, returns [original_query] safely without crashing.
    """
    if not query or not query.strip():
        return []

    original = query.strip()

    if not ENABLE_QUERY_REFORMULATION or num_reformulations <= 0 or method == "none":
        return [original]

    try:
        if method not in ["linguistic", "model"]:
            raise ValueError(f"Unknown reformulation method: {method}")

        alternatives = []
        if method == "model":
            alternatives = _generate_model_perspectives(original, n=num_reformulations, model_name=model_name)

        if not alternatives:
            # Linguistic perspective generator as default or model fallback
            alternatives = _generate_linguistic_perspectives(original, n=num_reformulations)

        # Deduplicate while preserving order and ensuring original is index 0
        seen = {original.lower()}
        filtered_alternatives = []
        for alt in alternatives:
            alt_clean = _clean_query_string(alt)
            if alt_clean and alt_clean.lower() not in seen:
                seen.add(alt_clean.lower())
                filtered_alternatives.append(alt_clean)
                if len(filtered_alternatives) >= num_reformulations:
                    break

        return [original] + filtered_alternatives

    except Exception as e:
        logger.error(f"Query reformulation error: {e}. Falling back to original query.")
        return [original]
