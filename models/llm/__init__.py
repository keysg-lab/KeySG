"""LLM/VLM interfaces for scene understanding."""

from models.llm.schemas import (
    FloorSummaryOutput,
    FunctionalTag,
    ImageDescription,
    ObjectDescription,
    ObjectTag,
    RoomBrief,
    SceneObjectSummary,
    SceneSummary,
)
from models.llm.prompts import (
    system_instruction_floor_summary,
    system_instruction_functional_tagging,
    system_instruction_grounded_description,
    system_instruction_grounding,
    system_instruction_per_frame,
    system_instruction_summary,
    system_instruction_tagging,
)
from models.llm.openai_api import GPTInterface
from models.llm.ollama_genai import OllamaGenAI, GenerationParams
from models.llm.gpt_vlm import GPT_VLMInterface
from models.llm.ollama_vlm import OllamaVLMInterface

__all__ = [
    # Schemas
    "FloorSummaryOutput",
    "FunctionalTag",
    "ImageDescription",
    "ObjectDescription",
    "ObjectTag",
    "RoomBrief",
    "SceneObjectSummary",
    "SceneSummary",
    # Prompts
    "system_instruction_floor_summary",
    "system_instruction_functional_tagging",
    "system_instruction_grounded_description",
    "system_instruction_grounding",
    "system_instruction_per_frame",
    "system_instruction_summary",
    "system_instruction_tagging",
    # API Interfaces
    "GPTInterface",
    "OllamaGenAI",
    "GenerationParams",
    # VLM Interfaces
    "GPT_VLMInterface",
    "OllamaVLMInterface",
]
