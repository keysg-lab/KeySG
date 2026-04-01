"""Data types for scene description results."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FrameVLMResult:
    """VLM result for a single frame."""
    index: int
    node_tags: List[str]
    description: Dict[str, Any]
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "path": self.path,
            "node_tags": self.node_tags,
            "description": self.description,
        }


@dataclass
class RoomVLMResult:
    """VLM result for a room (aggregated from frames)."""
    id: str
    frames: List[FrameVLMResult]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "frames": [f.to_dict() for f in self.frames],
            "summary": self.summary,
        }
