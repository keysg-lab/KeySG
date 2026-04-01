"""Node extraction and merging for 3D scene graph construction."""

import hashlib
import os
import pickle
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keysg.scene_segmentor.obj_node import ObjNode
from keysg.utils.clip_utils import CLIPFeatureExtractor
from keysg.utils.img_utils import crop_image, get_mask_score
from keysg.utils.pcd_utils import (
    compute_3d_bbox_iou,
    find_overlapping_ratio_faiss,
    pcd_denoise_dbscan,
)
from keysg.utils.vis_utils import (
    visualize_functional_elements,
    visualize_nodes_collection,
    visualize_single_node,
)
from models.gsam2.gsam2 import GroundingSAM2
from models.llm.gpt_vlm import GPT_VLMInterface

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class NodesRepo:
    """Repository for extracting, merging, and managing 3D object nodes from RGB-D sequences."""

    _shared_segmentor: Optional[Any] = None
    _shared_clip: Optional[CLIPFeatureExtractor] = None

    def __init__(
        self,
        dataset: Any,
        clip_config: Dict[str, Any],
        gsam2_config: Optional[Dict[str, Any]] = None,
        tags: str = "",
        fun_tags: str = "",
        selected_frame_indices: Optional[List[int]] = None,
        functional_elements_method: Optional[str] = None,
        output_dir: str = "",
    ):
        self.dataset = dataset
        self.clip_config = clip_config
        self.gsam2_config = gsam2_config or {}
        self.tags = tags
        self.fun_tags = fun_tags
        self.selected_frame_indices = selected_frame_indices or []
        self.output_dir = output_dir

        self.object_nodes: List[ObjNode] = []
        self.fun_nodes: List[ObjNode] = []
        self.gsam2: Optional[GroundingSAM2] = None
        self.clip_extractor: Optional[CLIPFeatureExtractor] = None
        self.vlm: Optional[GPT_VLMInterface] = None
        self._functional_elements_detector: Optional[GroundingSAM2] = None

        # Validate and store functional elements method
        method = (functional_elements_method or "sparse_tags").strip().lower()
        if method not in ("sparse_tags", "detector", "dense_tags"):
            logger.warning(
                f"Unknown functional_elements_method='{functional_elements_method}', "
                "falling back to 'sparse_tags'"
            )
            method = "sparse_tags"
        self.functional_elements_method = method

        self._initialize_components()

    def reconfigure_for_room(
        self,
        selected_frame_indices: List[int],
        tags: str = "",
        fun_tags: str = "",
        functional_elements_method: Optional[str] = None,
        output_dir: str = "",
    ) -> "NodesRepo":
        """Reconfigure for a new room without reloading models."""
        self.selected_frame_indices = selected_frame_indices
        self.tags = tags
        self.fun_tags = fun_tags
        if functional_elements_method:
            method = functional_elements_method.strip().lower()
            if method in ("sparse_tags", "detector", "dense_tags"):
                self.functional_elements_method = method
            else:
                logger.warning(
                    "Unknown functional_elements_method '%s' during reconfigure; "
                    "keeping '%s'",
                    functional_elements_method,
                    self.functional_elements_method,
                )
        if output_dir:
            self.output_dir = output_dir
        self.object_nodes = []
        self.fun_nodes = []
        return self

    def _initialize_components(self) -> None:
        """Initialize shared model instances."""
        # Initialize VLM for functional element tagging
        self.vlm = GPT_VLMInterface()

        if NodesRepo._shared_segmentor is None:
            logger.info("[NodesRepo] Loading GroundingSAM2...")
            cfg = self.gsam2_config
            detection_mode = (cfg.get("detection_mode") or "llmdet").strip().lower()
            gsam_kwargs = self._build_gsam_kwargs(cfg, detection_mode)
            NodesRepo._shared_segmentor = GroundingSAM2(**gsam_kwargs)
        self.gsam2 = NodesRepo._shared_segmentor

        if NodesRepo._shared_clip is None:
            logger.info("[NodesRepo] Loading CLIP extractor...")
            NodesRepo._shared_clip = CLIPFeatureExtractor(dict(self.clip_config))
        self.clip_extractor = NodesRepo._shared_clip

    def _build_gsam_kwargs(self, cfg: Dict, detection_mode: str) -> Dict[str, Any]:
        """Build kwargs for GroundingSAM2 initialization."""
        kwargs = {
            "detection_mode": detection_mode,
            "sam2_checkpoint": cfg.get(
                "sam2_checkpoint", "./checkpoints/sam2.1_hiera_large.pt"
            ),
            "sam2_model_config": cfg.get(
                "sam2_model_config", "sam2.1/sam2.1_hiera_l.yaml"
            ),
            "ram_pretrained": cfg.get(
                "ram_pretrained", "./checkpoints/ram_plus_swin_large_14m.pth"
            ),
            "ram_image_size": cfg.get("ram_image_size", 384),
        }

        model_id_map = {
            "grounding_dino": (
                "grounding_model_id",
                "IDEA-Research/grounding-dino-base",
            ),
            "owlv2": ("owlv2_model_id", "google/owlv2-base-patch16-ensemble"),
            "llmdet": ("llmdet_model_id", "iSEE-Laboratory/llmdet_large"),
        }

        if detection_mode in model_id_map:
            key, default = model_id_map[detection_mode]
            kwargs[key] = cfg.get(key, default)
        else:
            logger.warning(
                f"Unknown detection_mode='{detection_mode}', defaulting to 'llmdet'"
            )
            kwargs["detection_mode"] = "llmdet"
            kwargs["llmdet_model_id"] = cfg.get(
                "llmdet_model_id", "iSEE-Laboratory/llmdet_large"
            )

        return kwargs

    def extract_initial_nodes(
        self,
        skip_frames: int = 10,
        max_frames: int = 500,
        box_threshold: float = 0.4,
    ) -> List[ObjNode]:
        """Extract initial object nodes from frames without merging."""
        logger.info("Extracting initial object nodes...")
        self.object_nodes = []

        max_frames = min(max_frames, len(self.dataset))
        frame_indices = (
            self.selected_frame_indices[::skip_frames]
            if self.selected_frame_indices
            else range(0, max_frames, skip_frames)
        )
        logger.info(
            f"Processing {len(list(frame_indices))} frames (skip={skip_frames})"
        )

        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            try:
                self._process_frame(frame_idx, box_threshold)
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")

        return self.object_nodes

    def _process_frame(self, frame_idx: int, box_threshold: float) -> None:
        """Process a single frame and extract object nodes."""
        rgb_image, depth_image, camera_pose = self.dataset[frame_idx]

        text_prompt = self.tags
        if not text_prompt:
            ram_tags = self.gsam2.tag_image(rgb_image)
            text_prompt = self.gsam2.ram_tags_to_prompt(ram_tags)

        results = self.gsam2.predict(
            image=rgb_image, text_prompt=text_prompt, box_threshold=box_threshold
        )

        if self.output_dir:
            output_path = os.path.join(
                self.output_dir, "detections", f"frame_{frame_idx:04d}_detections.png"
            )
            self.gsam2.visualize_results(
                results, rgb_image, visualize=False, output_path=output_path
            )

        boxes = results.get("boxes")
        if boxes is None or len(boxes) == 0:
            return

        for i in range(len(boxes)):
            node = self._create_node_from_detection(
                results, i, rgb_image, depth_image, camera_pose, frame_idx
            )
            if node:
                self.object_nodes.append(node)

    def _create_node_from_detection(
        self,
        results: Dict,
        idx: int,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_pose: np.ndarray,
        frame_idx: int,
    ) -> Optional[ObjNode]:
        """Create an ObjNode from a detection result."""
        bbox_2d = results["boxes"][idx]

        # Skip near-full-image bboxes
        h, w = rgb_image.shape[:2]
        if (
            bbox_2d[0] < 0.1 * w
            and bbox_2d[1] < 0.1 * h
            and bbox_2d[2] > 0.9 * w
            and bbox_2d[3] > 0.9 * h
        ):
            return None

        mask_2d = results.get("masks", [None] * (idx + 1))[idx]
        label = (
            results.get("labels", [f"object_{idx}"])[idx]
            if "labels" in results
            else f"object_{idx}"
        )

        if mask_2d is None:
            return None

        pcd_3d = self.dataset.project_2d_mask_to_3d(
            mask_2d, depth_image, rgb_image, camera_pose
        )
        if pcd_3d is None or pcd_3d.is_empty():
            return None

        pcd_3d = pcd_3d.voxel_down_sample(voxel_size=0.01)
        pcd_3d, _ = pcd_3d.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

        if pcd_3d.is_empty():
            return None

        return ObjNode(
            id=f"obj_{idx}_{uuid.uuid4().hex[:3]}",
            bbox_3d=[],
            bboxs_2d=[bbox_2d],
            label=label,
            pcd=pcd_3d,
            feature=np.array([]),
            masks_2d=[mask_2d],
            rgb_frames=[rgb_image],
            frame_indices=[frame_idx],
        )

    def process_and_merge_nodes(
        self,
        nodes: List[ObjNode],
        similarity_threshold: float = 1.2,
        radius: float = 0.02,
        post_hoc_iou_thresh: float = 0.05,
        post_hoc_sim_thresh: float = 0.8,
        extract_functional_elements: bool = True,
    ) -> List[ObjNode]:
        """Merge and enrich object nodes through spatial merging, denoising, and feature extraction."""
        if not nodes:
            logger.warning("No object nodes provided.")
            return []

        logger.info(f"Starting merge with {len(nodes)} nodes...")

        merged = self._greedy_merge_nodes(nodes.copy(), similarity_threshold, radius)
        merged = self._post_hoc_merge_nodes(
            merged, post_hoc_iou_thresh, post_hoc_sim_thresh, radius
        )
        self._denoise_nodes(merged)

        torch.cuda.empty_cache()

        # Initialize functional elements segmentation tools
        if extract_functional_elements:
            self._init_functional_elements_segmentation()

        self._select_best_masks(merged)

        # Segment and merge functional elements
        if extract_functional_elements:
            fun_nodes = self._segment_and_merge_functional_elements(
                merged, post_hoc_iou_thresh, post_hoc_sim_thresh, radius
            )
            self._denoise_nodes(fun_nodes)

            # Assign functional elements back to their objects
            merged = self._assign_functional_elements_to_objects(
                fun_nodes, merged, similarity_threshold, radius
            )

        # Another round of post-hoc merging
        merged = self._post_hoc_merge_nodes(
            merged, post_hoc_iou_thresh, post_hoc_sim_thresh, radius
        )
        self._extract_clip_features(merged)
        self._compute_bboxes(merged)

        logger.info(f"Merge complete. Final node count: {len(merged)}")
        self.object_nodes = merged
        return merged

    def _greedy_merge_nodes(
        self,
        nodes: List[ObjNode],
        similarity_threshold: float,
        radius: float,
    ) -> List[ObjNode]:
        """Greedily merge nodes by geometric similarity."""
        merged: List[ObjNode] = []
        node_counts: Dict[str, int] = {}

        for node in tqdm(nodes, desc="Merging nodes"):
            best_match, best_sim = None, 0.0

            for existing in merged:
                sim = self._compute_geometric_similarity(node, existing, radius)
                if sim > best_sim:
                    best_sim, best_match = sim, existing

            threshold = (
                similarity_threshold * 0.25
                if self._is_structural(node)
                else similarity_threshold
            )

            if best_match and best_sim > threshold:
                self._fuse_nodes(best_match, node, node_counts)
            else:
                merged.append(node)
                node_counts[node.id] = 1

        logger.info(f"Greedy merge: {len(nodes)} -> {len(merged)} nodes")
        return merged

    def _is_structural(self, node: ObjNode) -> bool:
        """Check if node is a structural element (wall, floor, window)."""
        return any(kw in node.label.lower() for kw in ["wall", "floor", "window"])

    def _post_hoc_merge_nodes(
        self,
        nodes: List[ObjNode],
        iou_threshold: float,
        similarity_threshold: float,
        radius: float,
    ) -> List[ObjNode]:
        """Merge nodes with high IoU and feature similarity."""
        if len(nodes) < 2:
            return nodes

        n = len(nodes)
        iou_matrix = np.zeros((n, n))

        for i in tqdm(range(n), desc="Computing IoU matrix"):
            for j in range(i + 1, n):
                iou = compute_3d_bbox_iou(nodes[i].pcd, nodes[j].pcd)
                iou_matrix[i, j] = iou_matrix[j, i] = iou

        pairs = [
            (i, j, iou_matrix[i, j])
            for i in range(n)
            for j in range(i + 1, n)
            if iou_matrix[i, j] > 0
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)

        kept = np.ones(n, dtype=bool)
        node_counts = {node.id: 1 for node in nodes}

        for i, j, iou in pairs:
            if iou <= iou_threshold:
                break
            if kept[i] and kept[j]:
                sim = self._compute_geometric_similarity(nodes[i], nodes[j], radius)
                if sim > similarity_threshold:
                    self._fuse_nodes(nodes[j], nodes[i], node_counts)
                    kept[i] = False

        result = [node for node, k in zip(nodes, kept) if k]
        logger.info(f"Post-hoc merge: {n} -> {len(result)} nodes")
        return result

    def _denoise_nodes(self, nodes: List[ObjNode]) -> None:
        """Denoise point clouds using DBSCAN."""
        for node in nodes:
            if node.pcd and len(node.pcd.points) > 0:
                node.pcd = pcd_denoise_dbscan(node.pcd, eps=0.01, min_points=50)

    def _init_functional_elements_segmentation(self) -> None:
        """Initialize tools needed for functional elements depending on configured method."""
        method = getattr(self, "functional_elements_method", "sparse_tags")
        if method == "detector":
            cfg = self.gsam2_config or {}
            fun_kwargs = {
                "detection_mode": "functional_elements",
                "sam2_checkpoint": cfg.get(
                    "sam2_checkpoint", "./checkpoints/sam2.1_hiera_large.pt"
                ),
                "sam2_model_config": cfg.get(
                    "sam2_model_config", "sam2.1/sam2.1_hiera_l.yaml"
                ),
                "fungraph_checkpoint": cfg.get(
                    "fungraph_checkpoint", "./checkpoints/fungraph_det.pt"
                ),
            }
            self._functional_elements_detector = GroundingSAM2(**fun_kwargs)
        elif method in ("sparse_tags", "dense_tags") and not self.fun_tags:
            if self.vlm is None:
                self.vlm = GPT_VLMInterface()

    def _segment_and_merge_functional_elements(
        self,
        nodes: List[ObjNode],
        post_hoc_iou_thresh: float,
        post_hoc_sim_thresh: float,
        radius: float,
    ) -> List[ObjNode]:
        """Segment functional elements for each node and post-merge them."""
        self.fun_nodes = []
        for node in tqdm(nodes, desc="Segmenting functional elements"):
            self._segment_functional_elements(node)

        logger.info(f"Extracted {len(self.fun_nodes)} functional elements.")

        self.fun_nodes = self._post_hoc_merge_nodes(
            self.fun_nodes,
            iou_threshold=post_hoc_iou_thresh,
            similarity_threshold=post_hoc_sim_thresh,
            radius=radius,
        )
        logger.info(f"Post-hoc merged to {len(self.fun_nodes)} functional elements.")
        return self.fun_nodes

    def _segment_functional_elements(
        self, node: ObjNode, box_threshold: float = 0.45
    ) -> List[ObjNode]:
        """Segment functional elements using the configured method."""
        if not node.rgb_frames:
            logger.warning(
                f"Node {node.id} has no RGB frame for functional element segmentation"
            )
            return self.fun_nodes

        best_rgb_frame = node.rgb_frames[0]
        best_frame_idx = node.frame_indices[0] if node.frame_indices else None

        if (
            self.functional_elements_method == "detector"
            and self._functional_elements_detector is not None
        ):
            func_results = self._functional_elements_detector.predict(
                image=best_rgb_frame,
                box_threshold=box_threshold,
                multimask_output=False,
            )
        else:
            if not self.fun_tags and self.vlm is not None:
                logger.info(
                    f"Tagging functional elements for node {node.id} using VLM..."
                )
                fun_tags_list = self.vlm.tag_functional_elements_in_image(
                    Image.fromarray(best_rgb_frame)
                )
                fun_tags = ". ".join(fun_tags_list)
            else:
                fun_tags = self.fun_tags

            func_results = self.gsam2.predict(
                image=best_rgb_frame,
                text_prompt=fun_tags,
                box_threshold=box_threshold,
                multimask_output=False,
            )
            logger.info(f"Functional element tags: {fun_tags}")

            if self.output_dir:
                func_output_dir = os.path.join(self.output_dir, "functional_elements")
                os.makedirs(func_output_dir, exist_ok=True)
                self.gsam2.visualize_results(
                    func_results,
                    best_rgb_frame,
                    visualize=False,
                    output_path=os.path.join(func_output_dir, f"frame_{node.id}.jpg"),
                )

        if (
            func_results
            and func_results.get("boxes") is not None
            and len(func_results["boxes"]) > 0
        ):
            for i in range(len(func_results["boxes"])):
                fun_mask = func_results["masks"][i]
                fun_label = (
                    func_results.get("labels", [])[i]
                    if func_results.get("labels")
                    else "unknown"
                )

                _, depth_image, camera_pose = self.dataset[best_frame_idx]
                fun_pcd = self.dataset.project_2d_mask_to_3d(
                    fun_mask, depth_image, best_rgb_frame, camera_pose
                )

                if not fun_pcd or len(fun_pcd.points) == 0:
                    continue

                node_id = f"fun_{uuid.uuid4().hex[:3]}"
                fun_bbox_2d = func_results["boxes"][i]
                fun_mask_2d = func_results["masks"][i]
                fun_node = ObjNode(
                    id=node_id,
                    bbox_3d=[],
                    bboxs_2d=[fun_bbox_2d],
                    label=fun_label,
                    pcd=fun_pcd if len(fun_pcd.points) > 0 else None,
                    masks_2d=[fun_mask_2d] if fun_mask_2d is not None else [],
                    rgb_frames=[best_rgb_frame] if fun_mask_2d is not None else [],
                    frame_indices=[best_frame_idx] if fun_mask_2d is not None else [],
                )
                self.fun_nodes.append(fun_node)

        return self.fun_nodes

    def _assign_functional_elements_to_objects(
        self,
        fun_nodes: List[ObjNode],
        obj_nodes: List[ObjNode],
        similarity_threshold: float,
        radius: float,
    ) -> List[ObjNode]:
        """Assign each functional element to the closest object by geometric similarity."""
        final_nodes = list(obj_nodes)

        for fun_node in fun_nodes:
            best_match = None
            best_similarity = 0.0

            for obj_node in obj_nodes:
                sim = self._compute_geometric_similarity(fun_node, obj_node, radius)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = obj_node

            if best_match is not None and best_similarity > similarity_threshold:
                logger.info(
                    f"Assigning functional element {fun_node.id} to object {best_match.id}"
                )
                if best_match.functional_elements is not None:
                    best_match.functional_elements.append(fun_node)
                else:
                    best_match.functional_elements = [fun_node]
            else:
                logger.info(
                    f"Functional element {fun_node.id} did not match any object."
                )
                final_nodes.append(fun_node)

        return final_nodes

    def _select_best_masks(self, nodes: List[ObjNode]) -> None:
        """Select best mask for each node."""
        for node in tqdm(nodes, desc="Selecting best masks"):
            self._select_best_mask(node)

    def _select_best_mask(self, node: ObjNode) -> Optional[int]:
        """Select the best mask for a node based on quality scores."""
        if (
            not node.masks_2d
            or not node.rgb_frames
            or len(node.masks_2d) != len(node.rgb_frames)
        ):
            return None

        if len(node.masks_2d) == 1:
            best_idx = 0
        else:
            best_idx, best_score = 0, -float("inf")
            for i, mask in enumerate(node.masks_2d):
                mask_tensor = (
                    torch.from_numpy(mask.astype(float))
                    if isinstance(mask, np.ndarray)
                    else mask
                )
                _, _, boundary_score, size_score = get_mask_score(mask_tensor)
                score = boundary_score + size_score
                if score > best_score:
                    best_score, best_idx = score, i

        node.masks_2d = [node.masks_2d[best_idx]]
        node.rgb_frames = [node.rgb_frames[best_idx]]
        node.frame_indices = (
            [node.frame_indices[best_idx]] if node.frame_indices else None
        )
        if node.bboxs_2d and len(node.bboxs_2d) > best_idx:
            node.bboxs_2d = [node.bboxs_2d[best_idx]]

        return best_idx

    def _extract_clip_features(self, nodes: List[ObjNode]) -> None:
        """Extract CLIP features for all nodes."""
        for node in tqdm(nodes, desc="Extracting CLIP features"):
            self._extract_clip_feature(node)

    def _extract_clip_feature(self, node: ObjNode) -> None:
        """Extract CLIP features for a single node."""
        if self.clip_extractor and node.label:
            try:
                node.text_feature = self.clip_extractor.get_text_feats(
                    [node.label.strip()]
                )[0]
            except Exception:
                node.text_feature = None

        if not node.rgb_frames or not node.bboxs_2d:
            return

        rgb_cropped = crop_image(node.rgb_frames[0], node.bboxs_2d[0])
        if rgb_cropped is None or rgb_cropped.size == 0:
            return
        node.best_crop = rgb_cropped
        node.feature = self.clip_extractor.get_img_feats(rgb_cropped)

    def _compute_bboxes(self, nodes: List[ObjNode]) -> None:
        """Compute 3D bounding boxes for nodes."""
        for node in nodes:
            if node.pcd and len(node.pcd.points) > 0:
                try:
                    node.bbox_3d = np.asarray(
                        node.pcd.get_oriented_bounding_box().get_box_points()
                    )
                except Exception:
                    node.bbox_3d = np.asarray(
                        node.pcd.get_axis_aligned_bounding_box().get_box_points()
                    )
            else:
                node.bbox_3d = None

    def _compute_geometric_similarity(
        self, node1: ObjNode, node2: ObjNode, radius: float
    ) -> float:
        """Compute geometric similarity using nearest neighbor ratio."""
        if (
            not node1.pcd
            or not node2.pcd
            or node1.pcd.is_empty()
            or node2.pcd.is_empty()
            or compute_3d_bbox_iou(node1.pcd, node2.pcd) < 0.1
        ):
            return 0.0
        return find_overlapping_ratio_faiss(node1.pcd, node2.pcd, radius)

    def _fuse_nodes(
        self, target: ObjNode, source: ObjNode, node_counts: Dict[str, int]
    ) -> None:
        """Fuse source node into target node."""
        # Merge labels
        labels = set()
        for node in [target, source]:
            if node.label:
                labels.update(s.strip() for s in node.label.split(",") if s.strip())
        target.label = ", ".join(sorted(labels)) if labels else target.label

        # Merge point clouds
        if source.pcd and not source.pcd.is_empty():
            if target.pcd and not target.pcd.is_empty():
                target.pcd = (target.pcd + source.pcd).voxel_down_sample(
                    voxel_size=0.01
                )
            else:
                target.pcd = source.pcd

        # Merge list attributes
        for attr in ["bboxs_2d", "masks_2d", "rgb_frames", "frame_indices"]:
            source_list = getattr(source, attr)
            if source_list:
                target_list = getattr(target, attr) or []
                setattr(target, attr, target_list + source_list)

        node_counts[target.id] = node_counts.get(target.id, 1) + 1

    def find_closest_objects(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[ObjNode, float]]:
        """Find closest objects to a text query using CLIP features."""
        if not self.object_nodes or not self.clip_extractor:
            return []

        query_features = self.clip_extractor.get_text_feats([query_text])
        results = []

        for node in self.object_nodes:
            if node.feature is not None:
                similarity = np.dot(query_features[0], node.feature)
                results.append((node, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save_nodes(self, filepath: str) -> None:
        """Save object nodes to individual pickle files."""
        os.makedirs(filepath, exist_ok=True)

        for i, node in enumerate(tqdm(self.object_nodes, desc="Saving nodes")):
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", node.id or f"node_{i}")
            if len(safe_id) > 80:
                h = hashlib.md5((node.id or "").encode("utf-8")).hexdigest()[:8]
                safe_id = f"{safe_id[:70]}_{h}"

            node_path = os.path.join(filepath, f"{safe_id}.pkl")
            with open(node_path, "wb") as f:
                pickle.dump(node.to_dict(), f)

        logger.info(f"Saved {len(self.object_nodes)} nodes to {filepath}")

    @staticmethod
    def load_nodes(filepath: str) -> List[ObjNode]:
        """Load object nodes from individual pickle files."""
        if not os.path.isdir(filepath):
            logger.error(f"Directory {filepath} does not exist")
            return []

        nodes = []
        for filename in tqdm(os.listdir(filepath), desc="Loading nodes"):
            if not filename.endswith(".pkl"):
                continue

            node_path = os.path.join(filepath, filename)
            try:
                if os.path.getsize(node_path) == 0:
                    continue
                with open(node_path, "rb") as f:
                    nodes.append(ObjNode.from_dict(pickle.load(f)))
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                logger.warning(f"Failed to load {filename}: {e}")

        logger.info(f"Loaded {len(nodes)} nodes from {filepath}")
        return nodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about object nodes."""
        if not self.object_nodes:
            return {"total_nodes": 0}

        labels = [node.label for node in self.object_nodes]
        unique_labels = set(labels)
        pcd_sizes = [
            len(n.pcd.points)
            for n in self.object_nodes
            if n.pcd and len(n.pcd.points) > 0
        ]

        return {
            "total_nodes": len(self.object_nodes),
            "unique_labels": len(unique_labels),
            "label_counts": {
                label: labels.count(label) for label in sorted(unique_labels)
            },
            "nodes_with_pcd": sum(
                1 for n in self.object_nodes if n.pcd and len(n.pcd.points) > 0
            ),
            "nodes_with_masks": sum(1 for n in self.object_nodes if n.masks_2d),
            "nodes_with_functional": sum(
                1 for n in self.object_nodes if n.functional_elements
            ),
            "avg_pcd_size": np.mean(pcd_sizes) if pcd_sizes else 0,
            "nodes_with_multiple_detections": sum(
                1 for n in self.object_nodes if n.bboxs_2d and len(n.bboxs_2d) > 1
            ),
        }

    @staticmethod
    def clear_fun_tags(tags: str, fun_tags: str) -> str:
        """Strip parts of functional tags that overlap with object tags."""
        if not tags:
            return fun_tags

        object_tags_set = {
            tag.strip().lower()
            for tag in tags.replace(".", ",").split(",")
            if tag.strip()
        }
        processed = []

        for fun_tag in (
            t.strip() for t in fun_tags.replace(".", ",").split(",") if t.strip()
        ):
            separator = "-" if "-" in fun_tag else " "
            words = fun_tag.lower().replace("-", " ").split()
            remaining = [w for w in words if w not in object_tags_set]
            if remaining:
                processed.append(separator.join(remaining))

        return ", ".join(processed)

    def visualize_object_node(
        self, node: ObjNode, window_name: Optional[str] = None
    ) -> None:
        """Visualize a single object node's point cloud and masks."""
        if window_name is None:
            window_name = f"Object: {node.label}"
        visualize_single_node(
            node=node, window_name=window_name, show_functional=True, show_bbox=True
        )

    def visualize_all_nodes(
        self,
        window_name: str = "All Object Nodes",
        show_functional: bool = True,
        show_bbox: bool = False,
    ) -> None:
        """Visualize all object nodes' point clouds together."""
        visualize_nodes_collection(
            nodes=self.object_nodes,
            window_name=window_name,
            show_functional=show_functional,
            show_bbox=show_bbox,
        )

    def visualize_functional_elements(
        self, window_name: str = "Functional Elements"
    ) -> None:
        """Visualize functional elements of all object nodes."""
        visualize_functional_elements(nodes=self.object_nodes, window_name=window_name)
