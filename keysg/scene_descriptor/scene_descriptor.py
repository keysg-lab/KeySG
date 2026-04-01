"""
SceneDescriptor: VLM-based room description pipeline.

Responsibilities:
- Run VLM to extract tags and descriptions for room frames
- Aggregate per-room summaries from frame observations
- Save and load VLM outputs
"""

from __future__ import annotations
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from .data_types import FrameVLMResult, RoomVLMResult
from .vlm_provider import create_vlm
from .utils import (
    coerce_valid_json,
    is_pcd_visible_in_frame,
    normalize_tags,
    parse_json_best_effort,
    sanitize_for_json,
)


class SceneDescriptor:
    """
    Run VLM across rooms and persist results.

    For each room, processes sparse_indices frames to generate:
    - Per-frame tags and descriptions
    - Aggregated room summary
    """

    def __init__(
        self,
        dataset: Any,
        output_dir: str = "output/pipeline",
        vlm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.output_dir = output_dir
        self.vlm_config = vlm_config or {}
        self.vlm = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_vlm(self) -> None:
        """Lazily initialize VLM."""
        if self.vlm is None:
            self.vlm = create_vlm(self.vlm_config)

    # ---- Tagging ----

    async def tag_rooms(
        self,
        rooms: List[Any],
        max_tags: int = 80,
        batch_size: int = 20,
    ) -> Dict[str, List[str]]:
        """Tag objects in all rooms using VLM batch processing."""
        images, room_ids = self._collect_room_images(rooms)
        if not images:
            return {}

        logger.info(f"Tagging {len(images)} frames across {len(rooms)} rooms")
        self._ensure_vlm()

        try:
            all_tags = await self.vlm.tag_objects_in_images_batch(
                images=images, max_tags=max_tags, batch_size=batch_size
            )
            return self._group_tags_by_room(rooms, room_ids, all_tags)
        except Exception as e:
            logger.error(f"Room tagging failed: {e}")
            return {r.id: [] for r in rooms}

    async def tag_functional_rooms(
        self,
        rooms: List[Any],
        max_tags: int = 80,
        batch_size: int = 20,
    ) -> Dict[str, List[str]]:
        """Tag functional elements in all rooms."""
        images, room_ids = self._collect_room_images(rooms)
        if not images:
            return {}

        logger.info(f"Functional tagging {len(images)} frames across {len(rooms)} rooms")
        self._ensure_vlm()

        try:
            all_tags = await self.vlm.tag_functional_elements_in_images_batch(
                images=images, max_tags=max_tags, batch_size=batch_size
            )
            return self._group_tags_by_room(rooms, room_ids, all_tags)
        except Exception as e:
            logger.error(f"Functional tagging failed: {e}")
            return {r.id: [] for r in rooms}

    def _collect_room_images(
        self, rooms: List[Any]
    ) -> Tuple[List[Image.Image], List[str]]:
        """Collect images from all room sparse_indices."""
        images, room_ids = [], []
        for room in rooms:
            for idx in room.sparse_indices:
                rgb, _, _ = self.dataset[idx]
                images.append(Image.fromarray(rgb))
                room_ids.append(room.id)
        return images, room_ids

    def _group_tags_by_room(
        self,
        rooms: List[Any],
        room_ids: List[str],
        all_tags: List[List[str]],
    ) -> Dict[str, List[str]]:
        """Group tags by room ID."""
        result = {r.id: set() for r in rooms}
        for i, tags in enumerate(all_tags):
            result[room_ids[i]].update(tags)
        return {rid: list(tags) for rid, tags in result.items()}

    # ---- Description ----

    async def describe_room(self, room: Any, batch_size: int = 20) -> RoomVLMResult:
        """Generate descriptions for a room."""
        frame_data, images, visible_nodes = self._prepare_room_frames(room)

        if not images:
            return RoomVLMResult(id=room.id, frames=[], summary="")

        logger.info(f"Describing {len(images)} frames for room {room.id}")
        self._ensure_vlm()

        try:
            descriptions = await self.vlm.describe_images_with_nodes_batch(
                images=images, visible_nodes_list=visible_nodes, batch_size=batch_size
            )

            frames = [
                FrameVLMResult(
                    index=frame_data[i]["index"],
                    node_tags=frame_data[i]["node_tags"],
                    description=coerce_valid_json(descriptions[i]),
                    path=frame_data[i]["path"],
                )
                for i in range(len(descriptions))
            ]

            summary = self._summarize_frames([f.description for f in frames])
            return RoomVLMResult(id=room.id, frames=frames, summary=summary)

        except Exception as e:
            logger.error(f"Failed to describe room {room.id}: {e}")
            return RoomVLMResult(id=room.id, frames=[], summary="")

    async def describe_rooms(
        self,
        rooms: List[Any],
        batch_size: int = 20,
    ) -> Dict[str, RoomVLMResult]:
        """Generate descriptions for multiple rooms concurrently."""
        logger.info(f"Describing {len(rooms)} rooms")

        tasks = [self.describe_room(r, batch_size) for r in rooms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for room, result in zip(rooms, results):
            if isinstance(result, RoomVLMResult):
                output[room.id] = result
            else:
                logger.error(f"Room {room.id} description failed: {result}")
                output[room.id] = RoomVLMResult(id=room.id, frames=[], summary="")

        return output

    async def describe_objects_with_vlm(
        self,
        room: Any,
        update_labels: bool = True,
        batch_size: int = 20,
    ) -> List[Any]:
        """Describe each object in a room via VLM using a red-bbox annotated frame.

        Annotates each node's best-view frame with a red bounding box so the VLM
        sees the target in its spatial context, then calls the VLM to get a
        refined label, description, attributes, affordances, state, and location.
        """
        objects = getattr(room, "objects", []) or []
        if not objects:
            logger.info("Room {} has no objects; skipping.", room.id)
            return []

        all_labels = [obj.label for obj in objects if obj.label]
        valid_indices, annotated_images, current_labels, nearby_labels_list = [], [], [], []

        for i, node in enumerate(objects):
            if not node.rgb_frames or not node.bboxs_2d:
                continue
            frame = node.rgb_frames[0]
            bbox = node.bboxs_2d[0]
            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                continue
            annotated = frame.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            valid_indices.append(i)
            annotated_images.append(Image.fromarray(annotated))
            current_labels.append(node.label or "")
            nearby_labels_list.append([lbl for j, lbl in enumerate(all_labels) if j != i and lbl])

        if not annotated_images:
            logger.warning("Room {}: no valid frames for VLM object description.", room.id)
            return objects

        logger.info("VLM-describing {} objects in room {}.", len(annotated_images), room.id)
        self._ensure_vlm()

        try:
            descriptions = await self.vlm.describe_object_in_context_batch(
                images=annotated_images,
                current_labels=current_labels,
                nearby_labels_list=nearby_labels_list,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error("VLM object description failed for room {}: {}", room.id, e)
            return objects

        for result_idx, node_idx in enumerate(valid_indices):
            desc = descriptions[result_idx]
            if not desc:
                continue
            node = objects[node_idx]
            node.vlm_description = desc
            if update_labels:
                refined = (desc.get("name") or "").strip()
                if refined:
                    node.label = refined

        logger.info("VLM description done for room {}.", room.id)
        return objects

    def describe_objects_from_keyframes(
        self,
        room: Any,
        room_result: RoomVLMResult,
        update_labels: bool = True,
    ) -> List[Any]:
        """Populate per-object descriptions by mining existing keyframe descriptions.

        Each FrameVLMResult produced by describe_room() contains an 'objects'
        list with entries keyed by node ID (generated by the grounded-description
        prompt). This method picks the highest-confidence entry per node across
        all frames — no additional VLM calls needed.
        """
        objects = getattr(room, "objects", []) or []
        if not objects:
            return []

        if not room_result or not room_result.frames:
            logger.warning("Room {}: no keyframe descriptions available.", room.id)
            return objects

        # Build node_id -> best ObjectDescription dict (highest confidence wins)
        best: Dict[str, Dict] = {}
        for frame in room_result.frames:
            frame_objects = (
                frame.description.get("objects", [])
                if isinstance(frame.description, dict)
                else []
            )
            for obj in frame_objects:
                oid = obj.get("id")
                if oid and obj.get("confidence", 0.0) > best.get(oid, {}).get("confidence", 0.0):
                    best[oid] = obj

        matched = 0
        for node in objects:
            desc = best.get(node.id)
            if desc is None:
                continue
            node.vlm_description = desc
            matched += 1
            if update_labels:
                refined = (desc.get("name") or "").strip()
                if refined:
                    node.label = refined

        logger.info("{}/{} objects matched from keyframes in room {}.", matched, len(objects), room.id)
        return objects

    def _prepare_room_frames(
        self, room: Any
    ) -> Tuple[List[Dict], List[Image.Image], List[Dict]]:
        """Prepare frame data for room description."""
        frame_data, images, visible_nodes = [], [], []

        for idx in room.sparse_indices:
            rgb, depth, pose = self.dataset[idx]
            rgb_path, _, _ = self.dataset.data_list[idx]

            # Find visible nodes
            nodes_in_frame = {}
            for node in room.objects:
                if is_pcd_visible_in_frame(
                    node.pcd, depth, pose,
                    self.dataset.depth_intrinsics,
                    self.dataset.depth_scale,
                ):
                    nodes_in_frame[node.id] = node.label

            frame_data.append({
                "index": idx,
                "path": rgb_path,
                "node_tags": list(nodes_in_frame.values()),
            })
            images.append(Image.fromarray(rgb))
            visible_nodes.append(nodes_in_frame)

        return frame_data, images, visible_nodes

    def _summarize_frames(self, descriptions: List[Dict[str, Any]]) -> str:
        """Summarize frame descriptions into room summary."""
        try:
            obs = [coerce_valid_json(d) for d in descriptions if isinstance(d, dict)]
            self._ensure_vlm()
            return self.vlm.summarize_scene(obs)
        except Exception as e:
            logger.error(f"Summarize failed: {e}")
            return ""

    # ---- Floor Summaries ----

    def summarize_floor(
        self,
        rooms: Sequence[Any],
        room_results: Optional[Dict[str, RoomVLMResult]] = None,
    ) -> Dict[str, Any]:
        """Summarize a floor from its rooms' summaries."""
        self._ensure_vlm()

        payload = []
        for room in rooms:
            summary = None
            if room_results and room.id in room_results:
                summary = room_results[room.id].summary

            if not summary:
                continue

            # Parse summary if it's JSON string
            if isinstance(summary, str):
                parsed = parse_json_best_effort(summary)
                summary = parsed if parsed else summary

            payload.append({
                "id": room.id,
                "room_type": getattr(room, "name", None),
                "summary": summary,
            })

        if not payload:
            return {"floor_caption": "", "rooms": []}

        try:
            result = self.vlm.summarize_floor(payload)
            parsed = parse_json_best_effort(result)
            return parsed if isinstance(parsed, dict) else {"floor_caption": "", "rooms": []}
        except Exception as e:
            logger.error(f"Floor summarization failed: {e}")
            return {"floor_caption": "", "rooms": []}

    async def summarize_floors(
        self,
        floors_and_rooms: Sequence[Tuple[Any, List[Any]]],
        room_results: Optional[Dict[str, RoomVLMResult]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Summarize multiple floors concurrently."""
        tasks, floor_ids = [], []

        for floor, rooms in floors_and_rooms:
            floor_id = getattr(floor, "floor_id", None)
            if floor_id is None:
                continue

            task = asyncio.to_thread(self.summarize_floor, rooms, room_results)
            tasks.append(task)
            floor_ids.append(str(floor_id))

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = {}
        for fid, result in zip(floor_ids, results):
            if isinstance(result, dict):
                summaries[fid] = result
            else:
                logger.error(f"Floor {fid} summarization failed: {result}")
                summaries[fid] = {"floor_caption": "", "rooms": []}

        return summaries

    # ---- Save/Load ----

    def save(
        self,
        rooms: List[Any],
        results: Dict[str, RoomVLMResult],
        base: Optional[str] = None,
    ) -> str:
        """Save room VLM outputs to disk."""
        base_dir = base or self.output_dir
        seg_dir = os.path.join(base_dir, "segmentation")
        scene_index = {"rooms": []}

        for room in rooms:
            result = results.get(room.id)
            if result is None:
                continue

            r_dir = os.path.join(seg_dir, f"floor_{room.floor_id}", f"room_{room.id}")
            os.makedirs(r_dir, exist_ok=True)

            payload = result.to_dict()
            if isinstance(payload.get("summary"), str):
                parsed = parse_json_best_effort(payload["summary"])
                if parsed:
                    payload["summary"] = parsed

            out_path = os.path.join(r_dir, f"room_{room.id}_vlm.json")
            with open(out_path, "w") as f:
                json.dump(sanitize_for_json(payload), f, indent=2)

            scene_index["rooms"].append({
                "id": room.id,
                "floor_id": room.floor_id,
                "vlm_path": out_path,
            })

        idx_path = os.path.join(base_dir, "scene_description_index.json")
        with open(idx_path, "w") as f:
            json.dump(scene_index, f, indent=2)

        return idx_path

    def save_floor_summaries(
        self,
        summaries: Dict[str, Dict[str, Any]],
        base: Optional[str] = None,
    ) -> str:
        """Save floor summaries to disk."""
        base_dir = base or self.output_dir
        os.makedirs(base_dir, exist_ok=True)

        # Scene-level file
        scene_path = os.path.join(base_dir, "floor_summaries.json")
        with open(scene_path, "w") as f:
            json.dump(sanitize_for_json(summaries), f, indent=2)

        # Per-floor files
        seg_dir = os.path.join(base_dir, "segmentation")
        for floor_id, payload in summaries.items():
            floor_dir = os.path.join(seg_dir, f"floor_{floor_id}")
            os.makedirs(floor_dir, exist_ok=True)
            with open(os.path.join(floor_dir, "floor_summary.json"), "w") as f:
                json.dump(sanitize_for_json(payload), f, indent=2)

        return scene_path

    @staticmethod
    def load_scene(base: str) -> Dict[str, RoomVLMResult]:
        """Load scene descriptions from disk."""
        idx_path = os.path.join(base, "scene_description_index.json")
        with open(idx_path) as f:
            index = json.load(f)

        results = {}
        for r in index.get("rooms", []):
            try:
                with open(r["vlm_path"]) as rf:
                    data = json.load(rf)

                frames = [
                    FrameVLMResult(
                        index=fd["index"],
                        node_tags=fd.get("node_tags", []),
                        description=fd.get("description", {}),
                        path=fd.get("path", f"frame_{fd['index']}.jpg"),
                    )
                    for fd in data.get("frames", [])
                ]

                results[data["id"]] = RoomVLMResult(
                    id=data["id"],
                    frames=frames,
                    summary=data.get("summary", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to load room VLM: {e}")

        return results
