"""Ollama-based VLM interface for scene understanding tasks."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from models.llm.ollama_genai import OllamaGenAI, GenerationParams
from models.llm._common import extract_tags_from_response, parse_json_best_effort
from models.llm.schemas import (
    FloorSummaryOutput,
    FunctionalTag,
    ImageDescription,
    ObjectCropDescription,
    ObjectTag,
    SceneSummary,
)
from models.llm.prompts import (
    system_instruction_floor_summary,
    system_instruction_functional_tagging,
    system_instruction_grounded_description,
    system_instruction_grounding,
    system_instruction_object_crop_description,
    system_instruction_per_frame,
    system_instruction_summary,
    system_instruction_tagging,
)


class OllamaVLMInterface:
    """VLM interface using Ollama for tagging, describing, and summarizing."""

    def __init__(self, model: str = "qwen3-vl:30b", text_model: Optional[str] = None) -> None:
        self.model = model
        self.text_model = text_model or model
        self.client = OllamaGenAI(default_model=self.model)
        self.text_client = (
            self.client if self.text_model == self.model
            else OllamaGenAI(default_model=self.text_model)
        )

    def tag_objects_in_image(self, image: Image.Image, max_tags: int = 100) -> List[str]:
        """Tag visible objects in an image."""
        prompt = f"List all distinct object categories present in this image. Cap at {max_tags} items."
        raw = self._vision(prompt, [image], response_schema=ObjectTag.model_json_schema())
        parsed = self._validate_json(ObjectTag, raw)
        if isinstance(parsed, ObjectTag):
            return parsed.tags[:max_tags]
        return extract_tags_from_response(raw, key_hint="tags", max_items=max_tags)

    def tag_functional_elements_in_image(self, image: Image.Image, max_tags: int = 50) -> List[str]:
        """Tag functional/interactive elements in an image."""
        prompt = f"List all functional control/interaction elements visible (e.g., handle, knob). Never include object names. Cap at {max_tags} items."
        raw = self._vision(prompt, [image], response_schema=FunctionalTag.model_json_schema())
        parsed = self._validate_json(FunctionalTag, raw)
        if isinstance(parsed, FunctionalTag):
            return parsed.functional_tags[:max_tags]
        return extract_tags_from_response(raw, key_hint="functional_tags", max_items=max_tags)

    def describe_image(self, image: Image.Image) -> Dict[str, Any]:
        """Describe an image for scene understanding."""
        prompt = "Describe this single RGB image for a scene understanding task."
        raw = self._vision(prompt, [image], response_schema=ImageDescription.model_json_schema())
        parsed = self._validate_json(ImageDescription, raw)
        if isinstance(parsed, ImageDescription):
            return parsed.model_dump(by_alias=True)
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return fallback
        except ValueError:
            pass
        return {"caption": None, "room_type_guess": None, "description": raw, "scene_layout": None, "objects": []}

    def describe_image_with_nodes(
        self,
        image: Image.Image,
        visible_nodes: Dict[str, str],
    ) -> Dict[str, Any]:
        """Describe an image with known 3D object nodes."""
        if not visible_nodes:
            return self.describe_image(image)

        node_info = json.dumps(visible_nodes, indent=2)
        prompt = (
            f"Describe this RGB image with visible 3D object nodes:\n\n"
            f"Visible Objects:\n{node_info}\n\n"
            f"Reference objects using their node ID. Describe spatial relationships."
        )
        raw = self._vision(
            prompt, [image],
            system_instruction=system_instruction_grounded_description(),
            response_schema=ImageDescription.model_json_schema(),
        )
        parsed = self._validate_json(ImageDescription, raw)
        if isinstance(parsed, ImageDescription):
            return parsed.model_dump(by_alias=True)
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return fallback
        except ValueError:
            pass
        return {"caption": None, "room_type_guess": None, "description": raw, "scene_layout": None, "objects": []}

    def summarize_scene(self, observations: Sequence[Union[Dict[str, Any], str]]) -> str:
        """Summarize multiple frame observations into a room-level description."""
        if not observations:
            return ""

        compact_obs = json.dumps(observations, ensure_ascii=False)
        prompt = (
            "Fuse these observations from the SAME room into a single room-level summary.\n"
            f"Observations: {compact_obs}\n\nProduce the comprehensive room-level scene summary."
        )
        raw = self._text(
            prompt,
            system_instruction=system_instruction_summary(),
            response_schema=SceneSummary.model_json_schema(),
        )
        parsed = self._validate_json(SceneSummary, raw)
        if isinstance(parsed, SceneSummary):
            return json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return json.dumps(fallback, ensure_ascii=False, indent=2)
        except ValueError:
            pass
        return raw

    def ground_summary(
        self,
        scene_summary: Union[str, Dict[str, Any]],
        detected_objects: List[str],
    ) -> Dict[str, Any]:
        """Ground a scene summary with detected object IDs."""
        if isinstance(scene_summary, str):
            try:
                summary_dict = json.loads(scene_summary)
            except json.JSONDecodeError:
                summary_dict = {"room_summary": scene_summary, "objects": []}
        else:
            summary_dict = scene_summary

        prompt = (
            f"Ground this scene summary with detected objects:\n\n"
            f"Scene Summary: {json.dumps(summary_dict, indent=2)}\n\n"
            f"Detected Objects: {json.dumps(detected_objects, indent=2)}\n\n"
            f"Match objects and assign appropriate IDs."
        )
        raw = self._text(
            prompt,
            system_instruction=system_instruction_grounding(),
            response_schema=SceneSummary.model_json_schema(),
        )
        parsed = self._validate_json(SceneSummary, raw)
        if isinstance(parsed, SceneSummary):
            return parsed.model_dump()
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return fallback
        except ValueError:
            pass
        return summary_dict

    def summarize_floor(self, rooms: Sequence[Union[Dict[str, Any], str]]) -> str:
        """Summarize a floor from multiple room summaries."""
        if not rooms:
            return json.dumps({"floor_caption": "", "rooms": []}, ensure_ascii=False)

        normalized = []
        for idx, item in enumerate(rooms):
            if isinstance(item, str):
                normalized.append({"id": None, "room_type": None, "room_summary": item, "index": idx})
            elif isinstance(item, dict):
                normalized.append({
                    "id": item.get("id"),
                    "room_type": item.get("room_type") or item.get("type") or item.get("name"),
                    "room_summary": item.get("room_summary") or item.get("summary") or item.get("description"),
                    "index": idx,
                })
            else:
                normalized.append({"id": None, "room_type": None, "room_summary": str(item), "index": idx})

        prompt = (
            f"Create a floor caption and short caption for each room.\n\n"
            f"Rooms: {json.dumps(normalized, ensure_ascii=False)}"
        )
        raw = self._text(
            prompt,
            system_instruction=system_instruction_floor_summary(),
            response_schema=FloorSummaryOutput.model_json_schema(),
        )
        parsed = self._validate_json(FloorSummaryOutput, raw)
        if isinstance(parsed, FloorSummaryOutput):
            return json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return json.dumps(fallback, ensure_ascii=False, indent=2)
        except ValueError:
            pass
        return json.dumps({"floor_caption": "", "rooms": []}, ensure_ascii=False)

    # Async batch methods
    async def tag_objects_in_images_batch(
        self, images: List[Image.Image], max_tags: int = 100, batch_size: int = 20
    ) -> List[List[str]]:
        return await self._run_batch(
            images, batch_size,
            lambda img: self.tag_objects_in_image(img, max_tags=max_tags),
            default=[],
        )

    async def tag_functional_elements_in_images_batch(
        self, images: List[Image.Image], max_tags: int = 50, batch_size: int = 20
    ) -> List[List[str]]:
        return await self._run_batch(
            images, batch_size,
            lambda img: self.tag_functional_elements_in_image(img, max_tags=max_tags),
            default=[],
        )

    async def describe_images_batch(
        self, images: List[Image.Image], batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        return await self._run_batch(
            images, batch_size,
            self.describe_image,
            default={"caption": None, "room_type_guess": None, "description": None, "scene_layout": None, "objects": []},
        )

    def describe_object_in_context(
        self,
        image: Image.Image,
        current_label: str = "",
        nearby_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Describe a single object highlighted by a red bbox in a full frame."""
        context = f"Current label of the target object (inside the red box): {current_label}."
        if nearby_labels:
            context += f" Other nearby objects in the room: {', '.join(nearby_labels)}."
        prompt = f"Describe the target object highlighted by the red bounding box.\n{context}"
        raw = self._vision(
            prompt,
            [image],
            system_instruction=system_instruction_object_crop_description(),
            response_schema=ObjectCropDescription.model_json_schema(),
        )
        parsed = self._validate_json(ObjectCropDescription, raw)
        if isinstance(parsed, ObjectCropDescription):
            return parsed.model_dump(by_alias=True)
        try:
            fallback = parse_json_best_effort(raw)
            if isinstance(fallback, dict):
                return fallback
        except ValueError:
            pass
        return {}

    async def describe_object_in_context_batch(
        self,
        images: List[Image.Image],
        current_labels: List[str],
        nearby_labels_list: List[List[str]],
        batch_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Batch-describe objects highlighted by red bboxes in full frames."""
        if not (len(images) == len(current_labels) == len(nearby_labels_list)):
            raise ValueError(
                "images, current_labels, and nearby_labels_list must have equal length"
            )

        async def call(idx: int) -> Dict[str, Any]:
            return self.describe_object_in_context(
                images[idx],
                current_label=current_labels[idx],
                nearby_labels=nearby_labels_list[idx],
            )

        return await self._run_batch(
            list(range(len(images))),
            batch_size,
            call,
            default={},
        )

    async def describe_images_with_nodes_batch(
        self,
        images: List[Image.Image],
        visible_nodes_list: List[Dict[str, str]],
        batch_size: int = 20,
    ) -> List[Dict[str, Any]]:
        if len(images) != len(visible_nodes_list):
            raise ValueError("Number of images must match number of visible_nodes entries")

        async def call(idx: int):
            return self.describe_image_with_nodes(images[idx], visible_nodes_list[idx])

        return await self._run_batch(
            list(range(len(images))), batch_size, call,
            default={"caption": None, "room_type_guess": None, "description": None, "scene_layout": None, "objects": []},
        )

    # Internal methods
    def _vision(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        *,
        system_instruction: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        params = GenerationParams(format=response_schema)
        return self.client.vision(prompt, images, params=params)

    def _text(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        params = GenerationParams(
            system_instruction=system_instruction,
            format=response_schema,
        )
        return self.text_client.text(prompt, params=params)

    @staticmethod
    def _validate_json(model_cls, raw: str):
        try:
            return model_cls.model_validate_json(raw)
        except Exception:
            return None

    @staticmethod
    async def _run_batch(items: List[Any], batch_size: int, fn, *, default: Any) -> List[Any]:
        results: List[Any] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            tasks = [asyncio.to_thread(fn, item) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_results:
                results.append(default if isinstance(res, Exception) else res)
        return results
