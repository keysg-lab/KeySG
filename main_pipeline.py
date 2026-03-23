"""Main pipeline for HovFun scene processing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import hydra
from loguru import logger
from omegaconf import DictConfig

from dataloader.clio import ClioDataset
from dataloader.fungraph3d import FunGraph3DDataset
from dataloader.hmp3d import HM3DSemDataset
from dataloader.replica import ReplicaDataset
from dataloader.scannet import ScanNetDataset
from dataloader.sfm import SFMDataset
from hovfun.graph.keysg_graph import KeySGGraph
from hovfun.scene_descriptor.scene_descriptor import SceneDescriptor
from hovfun.scene_segmentor.extract_nodes import NodesRepo
from hovfun.scene_segmentor.scene_segmentor import SceneSegmentor
from hovfun.utils.logging_setup import setup_logging
from hovfun.utils.vis_utils import (
    label_keyframe,
    project_objects_to_masks,
    match_detections_to_objects,
    draw_id_labels,
)
from models.gsam2.gsam2 import GroundingSAM2

hydra.core.global_hydra.GlobalHydra.instance().clear()

DATASET_REGISTRY = {
    "hm3d": HM3DSemDataset,
    "hmp3d": HM3DSemDataset,
    "hm3dsem": HM3DSemDataset,
    "fungraph3d": FunGraph3DDataset,
    "fungraph": FunGraph3DDataset,
    "fg3d": FunGraph3DDataset,
    "replica": ReplicaDataset,
    "clio": ClioDataset,
    "cliodataset": ClioDataset,
    "sfm": SFMDataset,
    "sfmdataset": SFMDataset,
    "scanet": ScanNetDataset,
    "scannet": ScanNetDataset,
}


class HovFunPipeline:
    """Main pipeline for scene segmentation, description, and node extraction."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset = None
        self.output_dir = None
        self.floor_rooms = None
        self.dense_map = None
        self.sampled_map = None
        self.room_descriptions = None
        self.object_nodes: List = []
        self.rooms: List = []
        self.floors: List = []
        self.keysg_graph: Optional[KeySGGraph] = None
        self._shared_nodes_repo: Optional[NodesRepo] = None

    def setup(self) -> None:
        """Initialize dataset and output directory."""
        ds_cfg = {
            "root_dir": self.cfg.dataset.root_dir,
            "depth_scale": self.cfg.dataset.depth_scale,
            "depth_min": self.cfg.dataset.depth_min,
            "depth_max": self.cfg.dataset.depth_max,
            "video_ids": ["video0"],
        }
        self.dataset = self._create_dataset(self.cfg.dataset.kind, ds_cfg)

        scene_group = f"{getattr(self.dataset, 'name', 'scene')}/{getattr(self.dataset, 'scene_name', 'default')}"
        self.output_dir = os.path.join(self.cfg.output_dir, scene_group)
        os.makedirs(self.output_dir, exist_ok=True)

        self._save_config()
        self._setup_logging()
        self.scene_descriptor = SceneDescriptor(
            dataset=self.dataset,
            output_dir=self.output_dir,
            vlm_config=getattr(self.cfg, "vlm", None),
        )

    def _save_config(self) -> None:
        """Save a copy of the run config to the output directory."""
        from omegaconf import OmegaConf

        config_path = os.path.join(self.output_dir, "config.yaml")
        try:
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            logger.info("Config saved to {}", config_path)
        except Exception as e:
            logger.warning("Failed to save config: {}", e)

    def _setup_logging(self) -> None:
        """Add per-scene file logging."""
        try:
            log_path = os.path.join(self.output_dir, "hovfun.log")
            logger.add(log_path, level="INFO", rotation="10 MB", retention=3)
            logger.info("Logging to {}", log_path)
        except Exception as e:
            logger.warning("Failed to add file logger: {}", e)

    def _create_dataset(self, kind: str, cfg: Dict[str, Any]):
        """Create dataset instance based on kind."""
        kind = (kind or "").strip().lower()
        dataset_cls = DATASET_REGISTRY.get(kind)
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset kind: {kind}")
        return dataset_cls(cfg)

    def load_scene_segmentation(self) -> None:
        """Load existing scene segmentation from disk."""
        logger.info("Loading scene segmentation from {}", self.output_dir)

        seg = SceneSegmentor(dataset=self.dataset, output_dir=self.output_dir)
        try:
            floors, floor_rooms = seg.load()
            dense_map, sampled_map = seg.get_room_pose_indices()

            if not dense_map or not sampled_map:
                logger.info("Regenerating pose assignments...")
                dense_map = seg._assign_poses_to_rooms(floor_rooms)
                sampled_map = seg._sample_representative_frames(floor_rooms, dense_map)

            self._store_segmentation_results(
                floors, floor_rooms, dense_map, sampled_map
            )
            self._save_keyframe_poses()
            logger.info(
                "Loaded {} floors with {} rooms",
                len(floors),
                sum(len(r) for _, r in floor_rooms),
            )

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Scene segmentation not found: {e}. Run segmentation first."
            )

    def load_scene_description(self) -> None:
        """Load existing scene description from disk."""
        logger.info("Loading scene description from {}", self.output_dir)

        try:
            self.room_descriptions = self.scene_descriptor.load_scene(self.output_dir)
            logger.info("Loaded descriptions for {} rooms", len(self.room_descriptions))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Scene description not found: {e}. Run description first."
            )

    def run_segmentation(self) -> None:
        """Run scene segmentation pipeline."""
        seg_cfg = self.cfg.segmentation
        seg = SceneSegmentor(
            dataset=self.dataset,
            output_dir=self.output_dir,
            fuse_every_k=seg_cfg.fuse_every_k,
            voxel_size=seg_cfg.voxel_size,
            grid_resolution=seg_cfg.grid_resolution,
            save_intermediate=seg_cfg.save_intermediate,
            sampling_eps=seg_cfg.sampling_eps,
            sampling_min_samples=seg_cfg.sampling_min_samples,
            sampling_rot_weight=seg_cfg.sampling_rot_weight,
            points_in_room_threshold=seg_cfg.points_in_room_threshold,
        )
        seg.run()
        logger.info("Segmentation saved to {}", seg.save())

        self._store_segmentation_results(
            seg.get_floors(),
            seg.get_rooms_by_floor(),
            *seg.get_room_pose_indices(),
        )
        self._save_keyframe_poses()

    def _store_segmentation_results(
        self,
        floors: List,
        floor_rooms: List,
        dense_map: Dict,
        sampled_map: Dict,
    ) -> None:
        """Store segmentation results in instance variables."""
        self.floors = floors
        self.floor_rooms = floor_rooms
        self.dense_map = dense_map
        self.sampled_map = sampled_map
        self.rooms = [room for _, rooms in floor_rooms for room in rooms]

    def _save_keyframe_poses(self) -> None:
        """Save keyframe camera poses to disk for use by the visualizer."""
        for floor, rooms in self.floor_rooms:
            for room in rooms:
                sparse = getattr(room, "sparse_indices", None)
                if not sparse:
                    continue
                fid = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))
                rid = getattr(room, "id", None)
                if not rid:
                    continue
                poses: Dict[str, Any] = {}
                for idx in sparse:
                    try:
                        _, _, pose = self.dataset[idx]
                        poses[str(idx)] = pose.tolist()
                    except Exception as e:
                        logger.warning("Could not get pose for frame {}: {}", idx, e)
                room_dir = os.path.join(
                    self.output_dir, "segmentation", f"floor_{fid}", f"room_{rid}"
                )
                os.makedirs(room_dir, exist_ok=True)
                poses_path = os.path.join(room_dir, "keyframe_poses.json")
                with open(poses_path, "w") as f:
                    json.dump(poses, f)

    def load_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Load existing nodes from disk."""
        logger.info("Loading nodes from {}", self.output_dir)

        summary = {}
        all_nodes = []

        for floor, rooms in self.floor_rooms:
            for room in rooms:
                rid = getattr(room, "id", None)
                fid = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))
                if not rid:
                    continue

                nodes_dir = os.path.join(
                    self.output_dir,
                    "segmentation",
                    f"floor_{fid}",
                    f"room_{rid}",
                    "nodes",
                )
                if os.path.exists(nodes_dir):
                    room.objects = NodesRepo.load_nodes(nodes_dir)
                    all_nodes.extend(room.objects)
                    summary[rid] = {"count": len(room.objects), "path": nodes_dir}

        if not summary:
            raise RuntimeError(
                f"No nodes found in {self.output_dir}. Run node extraction first."
            )

        self.object_nodes = all_nodes
        logger.info("Loaded nodes for {} rooms", len(summary))
        return summary

    def run_node_extraction(self) -> None:
        """Run node extraction pipeline."""
        nodes_cfg = self.cfg.nodes
        self._extract_nodes_for_rooms(
            skip_frames=nodes_cfg.skip_frames,
            max_frames=None if nodes_cfg.max_frames <= 0 else nodes_cfg.max_frames,
            post_hoc_iou_thresh=nodes_cfg.post_hoc_iou_thresh,
            post_hoc_sim_thresh=nodes_cfg.post_hoc_sim_thresh,
            similarity_threshold=nodes_cfg.similarity_threshold,
            radius=nodes_cfg.radius,
        )
        self.object_nodes = [
            obj
            for _, rooms in self.floor_rooms
            for room in rooms
            for obj in room.objects
        ]

    def _extract_nodes_for_rooms(
        self,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        post_hoc_iou_thresh: float = 0.05,
        post_hoc_sim_thresh: float = 0.8,
        similarity_threshold: float = 1.2,
        radius: float = 0.02,
    ) -> None:
        """Extract, merge, and save object nodes for all rooms."""
        for floor, rooms in self.floor_rooms:
            for room in rooms:
                rid = getattr(room, "id", None)
                fid = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))
                if not rid or not room.indices:
                    continue

                room_tags = self._get_room_tags(rid, fid)
                repo = self._extract_nodes_for_room(
                    room=room,
                    floor_id=fid,
                    room_tags=room_tags,
                    skip_frames=skip_frames,
                    max_frames=max_frames,
                    post_hoc_iou_thresh=post_hoc_iou_thresh,
                    post_hoc_sim_thresh=post_hoc_sim_thresh,
                    similarity_threshold=similarity_threshold,
                    radius=radius,
                )
                self._save_room_nodes(repo, rid, fid)

    def _get_room_tags(self, rid: str, fid: str) -> List[str]:
        """Load or generate room tags."""
        room_dir = os.path.join(
            self.output_dir, "segmentation", f"floor_{fid}", f"room_{rid}"
        )
        tags_path = os.path.join(room_dir, "object_tags.json")

        if os.path.exists(tags_path) and self.cfg.nodes.object_tags == "vlm":
            with open(tags_path, "r") as f:
                return json.load(f).get("tags", [])

        if self.cfg.nodes.object_tags == "vlm":
            room = next(
                (
                    r
                    for _, rooms in self.floor_rooms
                    for r in rooms
                    if getattr(r, "id", None) == rid
                ),
                None,
            )
            if room:
                tags_dict = asyncio.run(self.scene_descriptor.tag_rooms([room]))
                tags = tags_dict.get(rid, [])
                os.makedirs(room_dir, exist_ok=True)
                with open(tags_path, "w") as f:
                    json.dump({"tags": list(tags)}, f)
                return tags

        return []

    def _extract_nodes_for_room(
        self,
        room,
        floor_id: str,
        room_tags: Optional[List[str]] = None,
        skip_frames: int = 1,
        max_frames: Optional[int] = None,
        post_hoc_iou_thresh: float = 0.05,
        post_hoc_sim_thresh: float = 0.8,
        similarity_threshold: float = 1.2,
        radius: float = 0.02,
    ) -> NodesRepo:
        """Extract and merge object nodes for a single room."""
        rid = room.id
        tags_txt = ". ".join(room_tags) if room_tags else ""

        # Choose frame pool: keyframes only or all dense frames
        use_keyframes_only = getattr(self.cfg.nodes, "use_keyframes_only", False)
        if use_keyframes_only:
            frame_indices = list(room.sparse_indices or room.indices)
            skip_frames = 1
            logger.info(
                "Extracting nodes for room {} ({} keyframes only)",
                rid,
                len(frame_indices),
            )
        else:
            frame_indices = room.indices
            logger.info(
                "Extracting nodes for room {} ({} dense frames, skip={})",
                rid,
                len(frame_indices),
                skip_frames,
            )

        # Get functional elements configuration
        fun_tags = getattr(self.cfg.nodes, "fun_tags", "")
        functional_elements_method = getattr(
            self.cfg.nodes, "functional_elements_method", "sparse_tags"
        )
        extract_functional_elements = getattr(
            self.cfg.nodes, "extract_functional_elements", True
        )

        room_output_dir = os.path.join(
            self.output_dir, "segmentation", f"floor_{floor_id}", f"room_{rid}"
        )

        if self._shared_nodes_repo is None:
            from hovfun.utils.clip_utils import DEFAULT_CLIP_CONFIG

            self._shared_nodes_repo = NodesRepo(
                dataset=self.dataset,
                tags=tags_txt,
                fun_tags=fun_tags,
                clip_config=dict(DEFAULT_CLIP_CONFIG),
                gsam2_config=getattr(self.cfg.nodes, "gsam2", {}),
                selected_frame_indices=frame_indices,
                functional_elements_method=functional_elements_method,
                output_dir=room_output_dir,
            )
        else:
            self._shared_nodes_repo.reconfigure_for_room(
                selected_frame_indices=frame_indices,
                tags=tags_txt,
                fun_tags=fun_tags,
                functional_elements_method=functional_elements_method,
                output_dir=room_output_dir,
            )

        repo = self._shared_nodes_repo
        nodes = repo.extract_initial_nodes(
            skip_frames=skip_frames,
            max_frames=max_frames or len(frame_indices),
        )
        room.objects = repo.process_and_merge_nodes(
            nodes,
            similarity_threshold=similarity_threshold,
            radius=radius,
            post_hoc_iou_thresh=post_hoc_iou_thresh,
            post_hoc_sim_thresh=post_hoc_sim_thresh,
            extract_functional_elements=extract_functional_elements,
        )

        logger.info("Extracted {} nodes for room {}", len(room.objects), rid)
        return repo

    def _save_room_nodes(self, repo: NodesRepo, rid: str, fid: str) -> None:
        """Save nodes for a room."""
        nodes_dir = os.path.join(
            self.output_dir, "segmentation", f"floor_{fid}", f"room_{rid}", "nodes"
        )
        if os.path.exists(nodes_dir):
            shutil.rmtree(nodes_dir)
        os.makedirs(nodes_dir, exist_ok=True)
        repo.save_nodes(nodes_dir)

    def build_keysg_graph(self, build_rag: bool = True) -> KeySGGraph:
        """Build the KeySG scene graph with optional RAG database."""
        logger.info("Building KeySG graph from {}", self.output_dir)

        rag_cfg = getattr(self.cfg, "rag", {})
        self.keysg_graph = KeySGGraph.from_output_dir(self.output_dir, build_rag=False)

        if build_rag:
            try:
                self.keysg_graph.build_rag_database(
                    embedding_model=getattr(
                        rag_cfg, "embedding_model", "text-embedding-3-small"
                    ),
                    compute_visual=getattr(rag_cfg, "compute_visual_embeddings", True),
                    use_cache=getattr(rag_cfg, "use_cache", True),
                )
                logger.info("KeySG graph with RAG built successfully")
            except Exception as e:
                logger.warning("Failed to build RAG: {}. Graph usable without RAG.", e)

        self.keysg_graph.save(os.path.join(self.output_dir, "keysg_graph.json"))
        return self.keysg_graph

    def query_scene(self, query_text: str, use_rag: bool = True) -> Dict[str, Any]:
        """Query the scene using natural language."""
        if self.keysg_graph is None:
            self.build_keysg_graph(build_rag=use_rag)

        if use_rag:
            result = self.keysg_graph.query(query_text)
            return {
                "query": result.query,
                "target_object_id": result.target_object_id,
                "target_label": (
                    result.target_object.label if result.target_object else None
                ),
                "room_id": result.room_id,
                "floor_id": result.floor_id,
                "confidence": result.confidence,
                "bbox_3d": result.bbox_3d,
                "metadata": result.metadata,
            }

        # Fallback to CLIP search via NodesRepo
        if self.object_nodes and self._shared_nodes_repo:
            matches = self._shared_nodes_repo.find_closest_objects(query_text, top_k=5)
            if matches:
                best, similarity = matches[0]
                return {
                    "query": query_text,
                    "target_object_id": best.id,
                    "target_label": best.label,
                    "confidence": float(similarity),
                    "all_matches": [(n.label, float(s)) for n, s in matches],
                }
        return {"query": query_text, "target_object_id": None}

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        logger.info("Starting HovFun pipeline...")
        self.setup()

        # Scene Segmentation
        if self.cfg.load.scene_segmentation:
            self.load_scene_segmentation()
        else:
            self.run_segmentation()

        # Node Extraction
        if self.cfg.load.nodes:
            self.load_nodes()
        else:
            self.run_node_extraction()

        # Scene Description
        if self.cfg.load.scene_description:
            self.load_scene_description()
        else:
            self._run_scene_description()

        # Per-object Descriptions (vlm or keyframe).
        # Runs after scene description so both methods have what they need.
        self._run_object_descriptions()

        # mark keyframes with object ids from llmdet and select object ids
        self._label_keyframes()

        # Build KeySG Graph
        if getattr(self.cfg, "build_rag", True):
            try:
                self.build_keysg_graph(build_rag=True)
            except Exception as e:
                logger.warning("Failed to build KeySG with RAG: {}", e)
                self.build_keysg_graph(build_rag=False)

        logger.info("Pipeline complete. Outputs: {}", self.output_dir)
        return {
            "output_dir": self.output_dir,
            "num_floors": len(self.floors),
            "num_rooms": len(self.rooms),
            "num_objects": len(self.object_nodes),
            "keysg_graph": self.keysg_graph,
        }

    def _label_keyframes(self) -> None:
        """Label keyframes by matching GSAM2 detections to projected 3D objects."""
        intrinsics = getattr(self.dataset, "rgb_intrinsics", None)
        if intrinsics is None:
            intrinsics = self.dataset.depth_intrinsics.copy()
            depth_h, depth_w = self.dataset.depth_H, self.dataset.depth_W
            rgb_h, rgb_w = self.dataset.rgb_H, self.dataset.rgb_W
            if depth_h != rgb_h or depth_w != rgb_w:
                sx, sy = rgb_w / depth_w, rgb_h / depth_h
                intrinsics[0, 0] *= sx
                intrinsics[0, 2] *= sx
                intrinsics[1, 1] *= sy
                intrinsics[1, 2] *= sy

        # Build a segmentor, reusing the shared instance if available
        if (
            self._shared_nodes_repo is not None
            and self._shared_nodes_repo.gsam2 is not None
        ):
            segmentor = self._shared_nodes_repo.gsam2
        else:
            cfg = getattr(self.cfg.nodes, "gsam2", {})
            segmentor = GroundingSAM2(
                    detection_mode="llmdet",
                    sam2_checkpoint=cfg.get(
                        "sam2_checkpoint", "./checkpoints/sam2.1_hiera_large.pt"
                    ),
                    sam2_model_config=cfg.get(
                        "sam2_model_config", "sam2_configs/sam2.1/sam2.1_hiera_l.yaml"
                    ),
                    llmdet_model_id=cfg.get(
                        "llmdet_model_id", "iSEE-Laboratory/llmdet_large"
                    ),
                )

        total_labeled = 0
        for floor, rooms in self.floor_rooms:
            for room in rooms:
                rid = getattr(room, "id", None)
                fid = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))
                if not rid or not room.sparse_indices or not room.objects:
                    continue

                out_dir = os.path.join(
                    self.output_dir,
                    "segmentation",
                    f"floor_{fid}",
                    f"room_{rid}",
                    "labeled_keyframes",
                )
                os.makedirs(out_dir, exist_ok=True)

                # Collect unique labels from all objects (split multi-labels on comma)
                all_labels = set()
                for obj in room.objects:
                    for part in str(getattr(obj, "label", "")).split(","):
                        tag = part.strip()
                        if tag:
                            all_labels.add(tag)
                text_prompt = ". ".join(sorted(all_labels)) + "."

                for idx in room.sparse_indices:
                    rgb, _, pose = self.dataset[idx]
                    h, w = rgb.shape[:2]

                    # 1) Project 3D object PCDs → 2D masks for visible objects
                    obj_masks = project_objects_to_masks(
                        room.objects,
                        pose,
                        intrinsics,
                        h,
                        w,
                    )
                    if not obj_masks:
                        continue

                    # 2) Run segmentor to get 2D detection masks
                    det_results = segmentor.predict(
                        image=rgb, text_prompt=text_prompt, box_threshold=0.2
                    )
                    if len(det_results["masks"]) == 0:
                        continue

                    # 3) Match each detection to the best-overlapping projected object
                    matches = match_detections_to_objects(
                        det_results["masks"], obj_masks
                    )

                    # 4) Draw only ID + label text (no masks or bboxes)
                    img = draw_id_labels(rgb, det_results["masks"], matches)

                    out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    total_labeled += 1

        logger.info("Labeled {} keyframes with object IDs", total_labeled)

    def _run_object_descriptions(self) -> None:
        """Describe each object in all rooms and save updated nodes to disk.

        Always mines per-object descriptions from keyframe results (free — scene
        description already produced them).  If object_desc.enabled is true, also
        runs a per-object VLM call (red-bbox annotated frame) which overwrites the
        keyframe-derived descriptions with more precise ones.
        """
        obj_desc_cfg = getattr(self.cfg.nodes, "object_desc", None)
        use_vlm = obj_desc_cfg and getattr(obj_desc_cfg, "enabled", False)
        update_labels = (
            getattr(obj_desc_cfg, "update_label", True) if obj_desc_cfg else True
        )
        batch_size = getattr(obj_desc_cfg, "batch_size", 20) if obj_desc_cfg else 20

        for floor, rooms in self.floor_rooms:
            for room in rooms:
                rid = getattr(room, "id", None)
                fid = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))
                if not rid or not getattr(room, "objects", None):
                    continue
                try:
                    # Step 1: always populate from keyframe descriptions (no extra VLM cost)
                    if self.room_descriptions:
                        updated = self.scene_descriptor.describe_objects_from_keyframes(
                            room,
                            room_result=self.room_descriptions.get(rid),
                            update_labels=update_labels,
                        )
                        self._persist_room_objects(updated, rid, fid)

                    # Step 2: optionally overwrite with dedicated per-object VLM calls
                    if use_vlm:
                        updated = asyncio.run(
                            self.scene_descriptor.describe_objects_with_vlm(
                                room, update_labels=update_labels, batch_size=batch_size
                            )
                        )
                        self._persist_room_objects(updated, rid, fid)
                except Exception as e:
                    logger.warning("Object description failed for room {}: {}", rid, e)

    def _persist_room_objects(self, nodes: List, rid: str, fid: str) -> None:
        """Overwrite a room's nodes on disk with the (updated) in-memory list."""
        nodes_dir = os.path.join(
            self.output_dir, "segmentation", f"floor_{fid}", f"room_{rid}", "nodes"
        )
        os.makedirs(nodes_dir, exist_ok=True)
        for i, node in enumerate(nodes):
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", node.id or f"node_{i}")
            if len(safe_id) > 80:
                h = hashlib.md5((node.id or "").encode("utf-8")).hexdigest()[:8]
                safe_id = f"{safe_id[:70]}_{h}"
            with open(os.path.join(nodes_dir, f"{safe_id}.pkl"), "wb") as f:
                pickle.dump(node.to_dict(), f)
        logger.info("Saved {} nodes for room {}.", len(nodes), rid)

    def _run_scene_description(self) -> None:
        """Run scene description for rooms and floors."""
        rooms_descriptions = asyncio.run(
            self.scene_descriptor.describe_rooms(self.rooms)
        )
        self.room_descriptions = rooms_descriptions
        self.scene_descriptor.save(self.rooms, rooms_descriptions, self.output_dir)

        floor_summaries = asyncio.run(
            self.scene_descriptor.summarize_floors(
                self.floor_rooms,
                room_results=rooms_descriptions,
            )
        )
        self.scene_descriptor.save_floor_summaries(
            floor_summaries, base=self.output_dir
        )
        logger.info("Saved floor summaries for {} floor(s)", len(floor_summaries))


@hydra.main(version_base=None, config_path="config", config_name="main_pipeline")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    setup_logging()
    pipeline = HovFunPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
