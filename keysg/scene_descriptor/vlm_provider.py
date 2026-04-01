"""VLM provider initialization and management."""

from __future__ import annotations
from typing import Any, Dict, Optional

from loguru import logger


def create_vlm(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a VLM interface based on configuration.

    Args:
        config: Dict with 'provider', 'model', and optionally 'text_model'
            - provider: 'openai', 'ollama', etc. (default: 'openai')
            - model: Model name (default: 'gpt-5-mini' for openai)
            - text_model: Text-only model for ollama

    Returns:
        VLM interface instance
    """
    cfg = config or {}
    provider = str(cfg.get("provider", "openai")).strip().lower()
    model = cfg.get("model")
    text_model = cfg.get("text_model")

    if provider in ("openai", "gpt", "openai_api"):
        from models.llm.gpt_vlm import GPT_VLMInterface
        return GPT_VLMInterface(model=model or "gpt-5-mini")

    if provider in ("ollama", "qwen3-vl", "qwen3"):
        from models.llm.ollama_vlm import OllamaVLMInterface
        return OllamaVLMInterface(model=model or "qwen3-vl", text_model=text_model)

    logger.warning(f"Unknown VLM provider '{provider}', defaulting to OpenAI")
    from models.llm.gpt_vlm import GPT_VLMInterface
    return GPT_VLMInterface(model=model or "gpt-5-mini")
