"""Query Analysis and Expansion utilities.

This module provides a helper function that uses the existing
`GPTInterface` (OpenAI API wrapper) to:

1. Identify the user's intent (locate single object, locate multiple objects, locate a room, multi-object task planning, or multi-level object retrieval) in a scene / graph-RAG context.
2. Extract key entity tokens (objects, locations, floors, room types, etc.).
3. Generate expanded / related search terms (synonyms, hypernyms, alternative phrasings).

Typical usage:

    from hovfun.rag.query_analysis import analyze_and_expand_query
    qa = analyze_and_expand_query("a place to put my drink on the first floor")
    print(qa)
    # qa.expanded_terms -> ["table", "coffee table", "countertop", "desk", "side table", ...]

You can then pass the expanded terms into your retriever by either:
    - Concatenating them to the original query
    - Averaging embeddings of each variant
    - Performing multiple searches and merging scores

The function is careful to enforce a *compact*, *machine-parseable* schema using
OpenAI structured output (Pydantic). If structured mode fails for any reason,
it falls back to a lightweight JSON instruction prompt and best‑effort parsing.
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import json
from pydantic import BaseModel, Field

import os

# Import GPTInterface - handle both installed package and direct execution scenarios
try:
    from models.llm.openai_api import GPTInterface
except ImportError:
    import sys
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from models.llm.openai_api import GPTInterface


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class QueryAnalysisResult:
    original_query: str
    target_object: Optional[str] = None
    anchor_objects: Optional[List[str]] = None
    relation_polarity: Optional[str] = None
    relation_degree: Optional[str] = None
    operator: Optional[str] = None
    behavior: Optional[str] = None
    status: Optional[str] = None

    @property
    def raw(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "target_object": self.target_object,
            "anchor_objects": self.anchor_objects,
            "relation_polarity": self.relation_polarity,
            "relation_degree": self.relation_degree,
            "operator": self.operator,
            "behavior": self.behavior,
            "status": self.status,
        }


# -----------------------------------------------------------------------------
# Pydantic schema for structured output
# -----------------------------------------------------------------------------
class _QuerySchema(BaseModel):  # type: ignore[misc]
    target_object: Optional[str] = Field(
        None, description="The main object being queried."
    )
    anchor_objects: Optional[List[str]] = Field(
        None, description="List of anchor objects referenced."
    )
    relation_polarity: Optional[str] = Field(
        None, description="Spatial relation polarity: 'near' or 'far'."
    )
    relation_degree: Optional[str] = Field(
        None,
        description="Degree of relation: 'positive', 'comparative', or 'superlative'.",
    )
    operator: Optional[str] = Field(
        None, description="Operator: 'argmin' for near, 'argmax' for far."
    )
    behavior: Optional[str] = Field(
        None,
        description="Behavior: 'comparative' ranks among same-class, 'superlative' returns global best.",
    )
    status: Optional[str] = Field(
        None,
        description="Status: 'ok', 'not_found_target', 'not_found_anchor', or 'parse_error'.",
    )


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = (
    "Role: You are a spatial query normalizer and selector for a 3D scene graph or object list.\n"
    "Inputs:\n"
    "- user_query: short imperative or declarative request (e.g., 'select the curtain that is closer to the nightstand').\n"
    "\n"
    "Normalization goals:\n"
    "- Extract target object, anchor objects, relation polarity (near vs far, adjacent, above, on, below, etc.), and degree (positive/comparative/superlative).\n"
    "- Resolve synonyms to canonical forms:\n"
    "  • Near-type: {near, close, nearby, next to, beside, adjacent to} → near\n"
    "  • Far-type: {far, far away} → far\n"
    "  • Comparative: {closer, nearer} → comparative_near; {farther, further} → comparative_far\n"
    "  • Superlative: {closest, nearest} → superlative_near; {farthest, furthest} → superlative_far\n"
    "- Interpret 'beside/next to/adjacent to' as near-type proximity; if an explicit adjacency predicate exists.\n"
    "\n"
    "Parsing:\n"
    "1) Target extraction: identify the requested object phrase (e.g., 'office chair' → class=office chair.\n"
    "2) Anchor resolution: identify anchor objects phrase (e.g., 'nightstand', 'door', etc.)\n"
    "3) Relation analysis:\n"
    "   - polarity: near | far\n"
    "   - degree: positive | comparative | superlative\n"
    "   - operator: argmin for near-family, argmax for far-family\n"
    "   - behavior: comparative ranks targets among same-class candidates and returns the best; superlative returns the global best among candidates.\n"
    "\n"
    "Ambiguity and errors:\n"
    "- If no target-object instances exist, set status='not_found_target'.\n"
    "- If no anchor-object instances exist (and no anchor ids provided), set status='not_found_anchor'.\n"
    "- If parsing fails to find target or anchor phrases, set status='parse_error'.\n"
)


def analyze_and_expand_query(
    query: str,
    *,
    model: str = "gpt-5-nano",
    client: Optional[Any] = None,
) -> QueryAnalysisResult:
    """Analyze a user query and generate expansion terms.

    Args:
        query: Raw user text.
        model: LLM model name (must support structured output for best path).
        client: Optional pre-instantiated `GPTInterface`.
        max_expanded_terms: Hard cap for expansion list trimming.

    Returns:
        QueryAnalysisResult object.
    """
    if GPTInterface is None:
        raise RuntimeError(
            "GPTInterface not available. Ensure OpenAI dependencies are installed."
        )
    gpt = client or GPTInterface()

    # First attempt: structured output
    structured = gpt.structured_prompt(  # type: ignore[attr-defined]
        prompt=f"User query: {query}",
        response_model=_QuerySchema,  # type: ignore[arg-type]
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        reasoning_effort="low", # NOTE: maybe "low" is sufficient.
    )

    query_parsed: Dict[str, Any] = {}
    if structured is not None:
        query_parsed = structured.model_dump()  # type: ignore[attr-defined]
    else:
        raise RuntimeError("Structured parsing failed.")

    return QueryAnalysisResult(
        original_query=query,
        target_object=query_parsed.get("target_object", None),
        anchor_objects=query_parsed.get("anchor_objects", None),
        relation_polarity=query_parsed.get("relation_polarity", None),
        relation_degree=query_parsed.get("relation_degree", None),
        operator=query_parsed.get("operator", None),
        behavior=query_parsed.get("behavior", None),
        status=query_parsed.get("status", "ok"),
    )


# -----------------------------------------------------------------------------
# Minimal CLI for ad-hoc testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    query = "choose the book that is over the trash can"
    result = analyze_and_expand_query(query)
    print("--- Query Analysis ---")
    print(json.dumps(result.raw, indent=2, ensure_ascii=False))
