"""Ollama GenAI interface for text and vision tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from PIL import Image

from models.llm._common import (
    ImageLike,
    encode_image_base64,
    parse_json_best_effort,
)

try:
    import ollama
except ImportError:
    ollama = None


@dataclass
class GenerationParams:
    """Generation parameters for Ollama API."""

    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_output_tokens: Optional[int] = 1024
    response_mime_type: Optional[str] = None  # Unused, kept for API compatibility
    system_instruction: Optional[str] = None
    seed: Optional[int] = None
    format: Optional[dict] = None  # JSON schema for structured outputs


class OllamaGenAI:
    """Thin wrapper around local Ollama API for text and vision."""

    def __init__(self, *, default_model: str = "gemma3:27b") -> None:
        if ollama is None:
            raise ImportError("The 'ollama' package is required. Install with: pip install ollama")
        self.default_model = default_model

    def text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Generate text for a single prompt."""
        messages: List[dict] = []
        if params and params.system_instruction:
            messages.append({"role": "system", "content": params.system_instruction})
        messages.append({"role": "user", "content": prompt})
        return self._generate(model=model or self.default_model, messages=messages, params=params)

    def vision(
        self,
        prompt: str,
        images: Sequence[ImageLike],
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """VLM call with one or more images."""
        image_b64 = [encode_image_base64(img) for img in images]
        messages: List[dict] = []
        if params and params.system_instruction:
            messages.append({"role": "system", "content": params.system_instruction})
        messages.append({"role": "user", "content": prompt, "images": image_b64})
        return self._generate(model=model or self.default_model, messages=messages, params=params)

    def text_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> Any:
        """Generate text and parse as JSON."""
        raw = self.text(prompt, model=model, params=params)
        return parse_json_best_effort(raw)

    def vision_json(
        self,
        prompt: str,
        images: Sequence[ImageLike],
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> Any:
        """VLM call and parse result as JSON."""
        raw = self.vision(prompt, images, model=model, params=params)
        return parse_json_best_effort(raw)

    def generate(
        self,
        contents: Sequence[Any],
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Generic multimodal call accepting text and images."""
        prompt_parts: List[str] = []
        images: List[ImageLike] = []
        for c in contents:
            if isinstance(c, str):
                if os.path.exists(c):
                    images.append(c)
                else:
                    prompt_parts.append(c)
            elif isinstance(c, (bytes, bytearray, Image.Image)):
                images.append(c)

        prompt = "\n".join(p for p in prompt_parts if p) or ""
        if images:
            return self.vision(prompt, images, model=model, params=params)
        return self.text(prompt, model=model, params=params)

    def kill(self) -> None:
        """Unload the model from memory."""
        try:
            ollama.generate(model=self.default_model, prompt="", keep_alive=0)
        except Exception:
            pass

    def _generate(
        self,
        *,
        model: str,
        messages: Sequence[dict],
        params: Optional[GenerationParams],
    ) -> str:
        """Internal generation method."""
        options: dict = {}
        fmt: Optional[dict] = None

        if params is not None:
            if params.temperature is not None:
                options["temperature"] = float(params.temperature)
            if params.top_k is not None:
                options["top_k"] = int(params.top_k)
            if params.top_p is not None:
                options["top_p"] = float(params.top_p)
            if params.max_output_tokens is not None:
                options["num_predict"] = int(params.max_output_tokens)
            if params.seed is not None:
                options["seed"] = int(params.seed)
            if params.format is not None:
                fmt = params.format

        try:
            resp = ollama.chat(
                model=model,
                messages=list(messages),
                options=options if options else None,
                format=fmt,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed: {e}") from e

        try:
            return (resp or {}).get("message", {}).get("content", "")
        except Exception:
            return ""
