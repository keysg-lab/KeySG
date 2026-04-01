"""Shared utilities for LLM modules."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, List, Sequence, Union

import numpy as np
from PIL import Image

ImageLike = Union[str, bytes, Image.Image, np.ndarray]


def parse_json_best_effort(text: str) -> Any:
    """Parse JSON from model response, handling markdown code blocks."""
    s = (text or "").strip()

    # Strip markdown code fences
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[7:].strip()
        elif s.startswith("```JSON"):
            s = s[7:].strip()
        else:
            s = s[3:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try extracting JSON array
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to parse JSON from model response")


def encode_image_base64(image: ImageLike, format: str = "PNG") -> str:
    """Encode an image to base64 string."""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    elif isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("utf-8")
    elif isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path does not exist: {image}")
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def encode_image_data_url(image: ImageLike, format: str = "jpeg") -> str:
    """Encode an image to a data URL for API calls."""
    b64 = encode_image_base64(image, format=format.upper())
    return f"data:image/{format.lower()};base64,{b64}"


def normalize_images_to_base64(images: Sequence[ImageLike]) -> List[str]:
    """Convert a sequence of images to base64 strings."""
    return [encode_image_base64(img) for img in images]


def extract_tags_from_response(
    raw: str,
    *,
    key_hint: str | None = None,
    max_items: int = 100,
) -> List[str]:
    """Extract a list of tags from a model response."""
    try:
        parsed = parse_json_best_effort(raw)
    except ValueError:
        parsed = raw

    tags: List[str] = []

    if isinstance(parsed, list):
        tags = [str(x).strip().lower() for x in parsed if isinstance(x, (str, int, float))]
    elif isinstance(parsed, dict):
        # Try key_hint first, then common keys
        for key in [key_hint, "tags", "objects", "functional_tags"]:
            if key and isinstance(parsed.get(key), list):
                tags = [str(x).strip().lower() for x in parsed[key]]
                break
    elif isinstance(parsed, str):
        tags = [t.strip().lower() for t in parsed.split(",") if t.strip()]

    # Deduplicate and cap
    seen = set()
    result: List[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
            if len(result) >= max_items:
                break
    return result
