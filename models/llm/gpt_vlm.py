"""GPT-based VLM interface for scene understanding tasks."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Sequence, Union

from PIL import Image

from models.llm.openai_api import GPTInterface
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


class GPT_VLMInterface:
    """VLM interface using GPTInterface for tagging, describing, and summarizing."""

    def __init__(self, model: str = "gpt-5-nano"):
        self.client = GPTInterface()
        self.model = model

    def tag_objects_in_image(self, image: Image.Image, max_tags: int = 100) -> List[str]:
        """Tag visible objects in an image."""
        prompt = f"List all distinct object categories present in this image. Cap at {max_tags} items."
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=ObjectTag,
                model=self.model,
                instructions=system_instruction_tagging(),
                image=image,
                reasoning_effort="low",
                detail="high",
            )
            if isinstance(response, ObjectTag):
                return response.tags[:max_tags]
        except Exception as e:
            print(f"Error in tag_objects_in_image: {e}")
        return []

    def tag_functional_elements_in_image(self, image: Image.Image, max_tags: int = 50) -> List[str]:
        """Tag functional/interactive elements in an image."""
        prompt = f"List all functional control/interaction elements visible (e.g., handle, knob). Never include object names. Cap at {max_tags} items."
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=FunctionalTag,
                model=self.model,
                instructions=system_instruction_functional_tagging(),
                image=image,
                reasoning_effort="low",
                detail="high",
            )
            if isinstance(response, FunctionalTag):
                return response.functional_tags[:max_tags]
        except Exception as e:
            print(f"Error in tag_functional_elements_in_image: {e}")
        return []

    def describe_image(self, image: Image.Image) -> Dict[str, Any]:
        """Describe an image for scene understanding."""
        prompt = "Describe this single RGB image for a scene understanding task."
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=ImageDescription,
                model=self.model,
                instructions=system_instruction_per_frame(),
                image=image,
                reasoning_effort="low",
                detail="high",
            )
            if isinstance(response, ImageDescription):
                return response.model_dump(by_alias=True)
        except Exception as e:
            print(f"Error in describe_image: {e}")
            try:
                raw = self.client.vision_prompt(
                    prompt, image=image, model=self.model,
                    instructions=system_instruction_per_frame()
                )
                return {"caption": None, "room_type_guess": None, "description": raw}
            except Exception:
                pass
        return {}

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
            f"Describe this RGB image with knowledge of visible 3D object nodes:\n\n"
            f"Visible Objects:\n{node_info}\n\n"
            f"Reference objects using their node ID. Describe spatial relationships."
        )
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=ImageDescription,
                model=self.model,
                instructions=system_instruction_grounded_description(),
                image=image,
                reasoning_effort="medium",
                detail="high",
            )
            if isinstance(response, ImageDescription):
                return response.model_dump(by_alias=True)
        except Exception as e:
            print(f"Error in describe_image_with_nodes: {e}")
            try:
                raw = self.client.vision_prompt(
                    prompt, image=image, model=self.model,
                    instructions=system_instruction_grounded_description()
                )
                return {"caption": None, "room_type_guess": None, "description": raw}
            except Exception:
                pass
        return {}

    def summarize_scene(self, observations: Sequence[Union[Dict[str, Any], str]]) -> str:
        """Summarize multiple frame observations into a room-level description."""
        if not observations:
            return ""

        compact_obs = json.dumps(observations, ensure_ascii=False)
        prompt = (
            "Fuse these observations from the SAME room into a single room-level summary.\n"
            f"Observations: {compact_obs}\n\nProduce the comprehensive room-level scene summary."
        )
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=SceneSummary,
                model="gpt-5-mini",
                instructions=system_instruction_summary(),
                reasoning_effort="medium",
            )
            if isinstance(response, SceneSummary):
                return json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error in summarize_scene: {e}")
            try:
                return self.client.text_prompt(
                    f"Summarize these room observations:\n\n{compact_obs}",
                    model="gpt-5-mini",
                    instructions=system_instruction_summary(),
                )
            except Exception:
                pass
        return ""

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
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=SceneSummary,
                model="gpt-5-mini",
                instructions=system_instruction_grounding(),
                reasoning_effort="medium",
            )
            if isinstance(response, SceneSummary):
                return response.model_dump()
        except Exception as e:
            print(f"Error in ground_summary: {e}")
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
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=FloorSummaryOutput,
                model="gpt-5-mini",
                instructions=system_instruction_floor_summary(),
                reasoning_effort="medium",
            )
            if isinstance(response, FloorSummaryOutput):
                return json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error in summarize_floor: {e}")
        return json.dumps({"floor_caption": "", "rooms": []}, ensure_ascii=False)

    # Async batch methods
    async def tag_objects_in_images_batch(
        self, images: List[Image.Image], max_tags: int = 100, batch_size: int = 20
    ) -> List[List[str]]:
        """Tag objects in multiple images using async batch processing."""
        return await self._batch_process(
            images, batch_size,
            lambda imgs: self._batch_tag_objects(imgs, max_tags),
            default=[],
        )

    async def tag_functional_elements_in_images_batch(
        self, images: List[Image.Image], max_tags: int = 50, batch_size: int = 20
    ) -> List[List[str]]:
        """Tag functional elements in multiple images using async batch processing."""
        return await self._batch_process(
            images, batch_size,
            lambda imgs: self._batch_tag_functional(imgs, max_tags),
            default=[],
        )

    async def describe_images_batch(
        self, images: List[Image.Image], batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Describe multiple images using async batch processing."""
        return await self._batch_process(
            images, batch_size,
            self._batch_describe,
            default={"caption": None, "room_type_guess": None, "description": None},
        )

    def describe_object_in_context(
        self,
        image: Image.Image,
        current_label: str = "",
        nearby_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Describe a single object highlighted by a red bbox in a full frame.

        Args:
            image: Full RGB frame with a red bounding box drawn around the target object.
            current_label: Current label/name of the target object (may be corrected by VLM).
            nearby_labels: Labels of other objects in the same room for spatial context.

        Returns:
            ObjectCropDescription as a dict, or {} on failure.
        """
        context = f"Current label of the target object (inside the red box): {current_label}."
        if nearby_labels:
            context += f" Other nearby objects in the room: {', '.join(nearby_labels)}."
        prompt = f"Describe the target object highlighted by the red bounding box.\n{context}"
        try:
            response = self.client.structured_prompt(
                prompt=prompt,
                response_model=ObjectCropDescription,
                model=self.model,
                instructions=system_instruction_object_crop_description(),
                image=image,
                reasoning_effort="low",
                detail="high",
            )
            if isinstance(response, ObjectCropDescription):
                return response.model_dump(by_alias=True)
        except Exception as e:
            print(f"Error in describe_object_in_context: {e}")
        return {}

    async def describe_object_in_context_batch(
        self,
        images: List[Image.Image],
        current_labels: List[str],
        nearby_labels_list: List[List[str]],
        batch_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Batch-describe objects highlighted by red bboxes in full frames.

        Args:
            images: Full RGB frames, each with a red bbox drawn around the target object.
            current_labels: Current label for each target object.
            nearby_labels_list: List of nearby object label lists (one per image).
            batch_size: VLM batch size.

        Returns:
            List of ObjectCropDescription dicts (empty dict on per-item failure).
        """
        if not (len(images) == len(current_labels) == len(nearby_labels_list)):
            raise ValueError(
                "images, current_labels, and nearby_labels_list must have equal length"
            )
        results: List[Dict[str, Any]] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_labels = current_labels[i : i + batch_size]
            batch_nearby = nearby_labels_list[i : i + batch_size]
            prompts = []
            for label, nearby in zip(batch_labels, batch_nearby):
                ctx = f"Current label of the target object (inside the red box): {label}."
                if nearby:
                    ctx += f" Other nearby objects in the room: {', '.join(nearby)}."
                prompts.append(
                    f"Describe the target object highlighted by the red bounding box.\n{ctx}"
                )
            batch_results = await self.client.structured_prompt_batch(
                prompts=prompts,
                response_model=ObjectCropDescription,
                model=self.model,
                images=batch_imgs,
                instructions=system_instruction_object_crop_description(),
                reasoning_effort="low",
                detail="high",
            )
            for r in batch_results:
                results.append(
                    r.model_dump(by_alias=True)
                    if isinstance(r, ObjectCropDescription)
                    else {}
                )
        return results

    async def describe_images_with_nodes_batch(
        self,
        images: List[Image.Image],
        visible_nodes_list: List[Dict[str, str]],
        batch_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Describe multiple images with nodes using async batch processing."""
        if len(images) != len(visible_nodes_list):
            raise ValueError("Number of images must match number of visible_nodes entries")

        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            batch_nodes = visible_nodes_list[i : i + batch_size]
            batch_results = await self._batch_describe_with_nodes(batch_imgs, batch_nodes)
            results.extend(batch_results)
        return results

    async def _batch_process(self, images, batch_size, processor, default):
        """Generic batch processing helper."""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            try:
                batch_results = await processor(batch)
                results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing error: {e}")
                results.extend([default] * len(batch))
        return results

    async def _batch_tag_objects(self, images: List[Image.Image], max_tags: int) -> List[List[str]]:
        prompts = [f"List all distinct object categories. Cap at {max_tags} items." for _ in images]
        results = await self.client.structured_prompt_batch(
            prompts=prompts, response_model=ObjectTag, model=self.model,
            images=images, instructions=system_instruction_tagging(),
            reasoning_effort="low", detail="high",
        )
        return [r.tags[:max_tags] if isinstance(r, ObjectTag) else [] for r in results]

    async def _batch_tag_functional(self, images: List[Image.Image], max_tags: int) -> List[List[str]]:
        prompts = [f"List all functional elements. Cap at {max_tags} items." for _ in images]
        results = await self.client.structured_prompt_batch(
            prompts=prompts, response_model=FunctionalTag, model=self.model,
            images=images, instructions=system_instruction_functional_tagging(),
            reasoning_effort="low", detail="high",
        )
        return [r.functional_tags[:max_tags] if isinstance(r, FunctionalTag) else [] for r in results]

    async def _batch_describe(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        prompts = ["Describe this single RGB image for scene understanding." for _ in images]
        results = await self.client.structured_prompt_batch(
            prompts=prompts, response_model=ImageDescription, model=self.model,
            images=images, instructions=system_instruction_per_frame(),
            reasoning_effort="low", detail="high",
        )
        default = {"caption": None, "room_type_guess": None, "description": None}
        return [r.model_dump(by_alias=True) if isinstance(r, ImageDescription) else default for r in results]

    async def _batch_describe_with_nodes(
        self, images: List[Image.Image], nodes_list: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        prompts = []
        for nodes in nodes_list:
            if not nodes:
                prompts.append("Describe this single RGB image for scene understanding.")
            else:
                node_info = json.dumps(nodes, indent=2)
                prompts.append(
                    f"Describe this RGB image with visible 3D nodes:\n\n{node_info}\n\n"
                    f"Reference objects using their node ID."
                )
        results = await self.client.structured_prompt_batch(
            prompts=prompts, response_model=ImageDescription, model=self.model,
            images=images, instructions=system_instruction_grounded_description(),
            reasoning_effort="medium", detail="high",
        )
        default = {"caption": None, "room_type_guess": None, "description": None}
        return [r.model_dump(by_alias=True) if isinstance(r, ImageDescription) else default for r in results]
