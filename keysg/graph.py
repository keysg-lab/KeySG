"""KeySG: Hierarchical Keyframe-Based 3D Scene Graph."""

from __future__ import annotations

import os
import json
import pickle
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from keysg.rag.graph_context_retriever import GraphContextRetriever
from keysg.scene_segmentor.obj_node import ObjNode


@dataclass
class FloorNode:
    id: str
    summary: str = ""
    rooms: List["RoomNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomNode:
    id: str
    floor_id: str
    summary: str = ""
    keyframes: List["KeyframeNode"] = field(default_factory=list)
    objects: List[ObjNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyframeNode:
    index: int
    room_id: str
    image_path: str
    labeled_image_path: str = ""
    description: str = ""
    object_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeySGGraph:
    """Hierarchical Keyframe-Based 3D Scene Graph."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.floors: List[FloorNode] = []
        self.rooms: Dict[str, RoomNode] = {}
        self.objects: Dict[str, ObjNode] = {}
        self._retriever: Optional[Any] = None
        self._rag_initialized: bool = False
        self.scene_name: str = ""
        self.dataset_name: str = ""
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def from_output_dir(cls, output_dir: str, build_rag: bool = True) -> "KeySGGraph":
        graph = cls(output_dir)
        graph._load_from_output_dir()
        if build_rag:
            graph.build_rag_database()
        return graph

    def _load_from_output_dir(self) -> None:
        if not self.output_dir or not os.path.isdir(self.output_dir):
            raise ValueError(f"Invalid output directory: {self.output_dir}")
        logger.info("Loading KeySG graph from {}", self.output_dir)
        self._load_scene_metadata()
        self._load_floor_summaries()
        self._load_rooms_and_objects()
        logger.info(
            "Loaded: {} floors, {} rooms, {} objects",
            len(self.floors), len(self.rooms), len(self.objects),
        )

    def _load_scene_metadata(self) -> None:
        index_path = os.path.join(self.output_dir, "segmentation", "index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                data = json.load(f)
            self.scene_name = data.get("scene_name", "")
            self.dataset_name = data.get("dataset_name", "")
            self.metadata = data

    def _load_floor_summaries(self) -> None:
        floor_path = os.path.join(self.output_dir, "floor_summaries.json")
        if not os.path.exists(floor_path):
            floor_path = os.path.join(self.output_dir, "segmentation", "floor_summaries.json")
        if os.path.exists(floor_path):
            with open(floor_path) as f:
                floor_data = json.load(f)
            for floor_id, data in floor_data.items():
                self.floors.append(FloorNode(id=floor_id, summary=data.get("floor_caption", ""), metadata=data))

    def _load_rooms_and_objects(self) -> None:
        seg_dir = os.path.join(self.output_dir, "segmentation")
        if not os.path.isdir(seg_dir):
            logger.warning("Segmentation directory not found: {}", seg_dir)
            return
        for floor_dir_name in os.listdir(seg_dir):
            if not floor_dir_name.startswith("floor_"):
                continue
            floor_path = os.path.join(seg_dir, floor_dir_name)
            if not os.path.isdir(floor_path):
                continue
            floor_id = floor_dir_name.replace("floor_", "")
            floor_node = next((f for f in self.floors if f.id == floor_id), None)
            if floor_node is None:
                floor_node = FloorNode(id=floor_id)
                self.floors.append(floor_node)
            for room_dir_name in os.listdir(floor_path):
                if not room_dir_name.startswith("room_"):
                    continue
                room_path = os.path.join(floor_path, room_dir_name)
                if not os.path.isdir(room_path):
                    continue
                room_id = room_dir_name.replace("room_", "")
                room_node = self._load_room(room_path, room_id, floor_id)
                if room_node:
                    floor_node.rooms.append(room_node)
                    self.rooms[room_id] = room_node

    def _load_room(self, room_path: str, room_id: str, floor_id: str) -> Optional[RoomNode]:
        room_node = RoomNode(id=room_id, floor_id=floor_id)

        # Build a per-object vlm description lookup from the room's VLM JSON.
        # frames[].description.objects[] entries are keyed by object id and used
        # to fill in vlm_description when it is absent from the pkl.
        vlm_obj_lookup: Dict[str, dict] = {}
        vlm_path = os.path.join(room_path, f"room_{room_id}_vlm.json")
        if not os.path.exists(vlm_path):
            vlm_path = os.path.join(room_path, f"{room_id}_vlm.json")
        if os.path.exists(vlm_path):
            with open(vlm_path) as f:
                vlm_data = json.load(f)
            summary = vlm_data.get("summary", {})
            room_node.summary = summary.get("room_summary", "") if isinstance(summary, dict) else str(summary or "")
            room_node.metadata = vlm_data
            labeled_dir = os.path.join(room_path, "labeled_keyframes")
            for frame_data in vlm_data.get("frames", []):
                idx = frame_data.get("index", 0)
                labeled_path = os.path.join(labeled_dir, f"frame_{idx:06d}.png")
                frame_desc = frame_data.get("description", {})
                frame_objects = frame_desc.get("objects", [])
                keyframe = KeyframeNode(
                    index=idx,
                    room_id=room_id,
                    image_path=frame_data.get("path", ""),
                    labeled_image_path=labeled_path if os.path.isfile(labeled_path) else "",
                    description=frame_desc.get("caption", ""),
                    object_ids=[o["id"] for o in frame_objects if o.get("id")],
                    metadata=frame_data,
                )
                room_node.keyframes.append(keyframe)
                # Collect per-object descriptions; first occurrence wins
                for obj_desc in frame_objects:
                    obj_id = obj_desc.get("id")
                    if obj_id and obj_id not in vlm_obj_lookup:
                        vlm_obj_lookup[obj_id] = obj_desc

        # Load ObjNode instances from pkl files.
        # Each pkl stores a plain dict matching ObjNode.to_dict() output.
        nodes_dir = os.path.join(room_path, "nodes")
        if os.path.isdir(nodes_dir):
            for node_file in sorted(os.listdir(nodes_dir)):
                if not node_file.endswith(".pkl"):
                    continue
                try:
                    with open(os.path.join(nodes_dir, node_file), "rb") as f:
                        raw = pickle.load(f)

                    if isinstance(raw, ObjNode):
                        obj_node = raw
                    elif isinstance(raw, dict):
                        obj_node = ObjNode.from_dict(raw)
                    else:
                        obj_node = ObjNode.from_dict(raw.__dict__ if hasattr(raw, "__dict__") else {})

                    # Supplement vlm_description from the JSON lookup when absent in pkl
                    if obj_node.vlm_description is None and obj_node.id in vlm_obj_lookup:
                        obj_node.vlm_description = vlm_obj_lookup[obj_node.id]

                    room_node.objects.append(obj_node)
                    self.objects[obj_node.id] = obj_node
                except Exception as e:
                    logger.warning("Failed to load object {}: {}", node_file, e)

        return room_node

    def build_rag_database(
        self,
        embedding_model: str = "text-embedding-3-small",
        compute_visual: bool = True,
        use_cache: bool = True,
    ) -> None:
        if GraphContextRetriever is None:
            raise RuntimeError("RAG dependencies not available")
        logger.info("Building RAG database")
        self._retriever = GraphContextRetriever(self.output_dir)
        self._retriever.build_chunks()
        self._retriever.compute_embeddings(
            model_name=embedding_model,
            use_cache=use_cache,
            compute_frame_visual=compute_visual,
            compute_object_visual=compute_visual,
        )
        self._retriever.build_faiss_index(use_cache=use_cache)
        self._rag_initialized = True
        logger.info("RAG database built successfully")

    def save(self, path: str) -> None:
        data = {
            "output_dir": self.output_dir,
            "scene_name": self.scene_name,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
            "floors": [f.id for f in self.floors],
            "rooms": list(self.rooms.keys()),
            "objects": list(self.objects.keys()),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved KeySG graph to {}", path)
