"""KeySG Scene Visualizer — unified Viser-based interactive viewer.

Shows the full KeySG scene graph in 3D:
  - Floor / room / object point clouds (per-instance colors, toggleable)
  - Camera frustums at world-space keyframe positions with RGB thumbnails
  - GUI panel for two query modes:
      1. Object grounding  — RAG + LLM → highlights the matched object + red 3D bbox
      2. Open-ended Q&A   — LLM answers arbitrary questions with reasoning

Usage:
    keysg-vis --scene_dir output/pipeline/ScanNet/scene0011_00
    python -m hovfun.visualization.visualizer --scene_dir <path> [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

import viser

from keysg.utils.load_utils import get_floors, get_objects, get_rooms, load_scene_nodes


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------


def _palette(n: int) -> np.ndarray:
    """Generate n perceptually distinct RGB colors (values in [0, 255])."""
    if n == 0:
        return np.empty((0, 3), dtype=np.uint8)
    golden = 0.618033988749895
    hues = (np.arange(n) * golden) % 1.0
    colors = []
    for h in hues:
        c = 0.72  # chroma
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = 0.9 - c
        if h < 1 / 6:
            rgb = (c, x, 0)
        elif h < 2 / 6:
            rgb = (x, c, 0)
        elif h < 3 / 6:
            rgb = (0, c, x)
        elif h < 4 / 6:
            rgb = (0, x, c)
        elif h < 5 / 6:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        colors.append([(v + m) * 255 for v in rgb])
    return np.array(colors, dtype=np.uint8)


def _pose_to_wxyz_pos(pose: np.ndarray):
    """Extract (wxyz quaternion, xyz position) from a 4×4 camera-to-world matrix."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    xyzw = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    return wxyz, t.astype(np.float32)


def _load_thumbnail(image_path: str, max_side: int = 320) -> Optional[np.ndarray]:
    """Load and downscale an RGB image for use as a frustum thumbnail."""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        logger.warning("Could not load thumbnail {}: {}", image_path, e)
        return None


# ---------------------------------------------------------------------------
# Grounding query helper (mirrors nr3d_eval._run_keysg_rag pipeline)
# ---------------------------------------------------------------------------

_OBJECT_SELECTION_SYSTEM_PROMPT = (
    "You are a spatial reasoning expert. Your PRIMARY task is to visually identify the object "
    "described by the USER QUERY by carefully examining the attached scene images.\n\n"
    "Each image has its FRAME_ID stamped in the top-left corner. Match this to the corresponding "
    "entry in 'Relevant Frames' to cross-reference the visual content with the text description.\n\n"
    "Decision Logic — follow in order:\n"
    "1. **Visual Grounding (primary):** Look at every attached image. Identify the object that best "
    "matches the USER QUERY in terms of appearance, color, shape, and position. Trust what you see.\n"
    "2. **Cross-Reference Candidate List (secondary):** The 'Target Object Candidates' list is a "
    "retrieval shortlist — use it to map your visual observation to an `object_id`. Do NOT blindly "
    "rank by list order; only pick an ID that visually matches what you found in step 1.\n"
    "3. **Spatial Verification:** Apply spatial constraints from the query (e.g., 'left of', 'near') "
    "using image evidence and any provided Spatial Relations.\n"
    "4. **Anchor Objects:** If anchor objects are listed, locate them visually first, then apply the "
    "spatial relation to narrow down the target.\n"
    "5. **Selection:** Choose the `object_id` with the strongest combined visual + spatial evidence. "
    "If ambiguous, provide the closest guess with low confidence.\n\n"
    "Output Requirements:\n"
    "- **Format:** Respond ONLY in the enforced JSON schema.\n"
    "- **ID Validity:** Only use IDs from the Candidate List — never hallucinate IDs.\n"
    "- **Confidence:**\n"
    "  - High (~0.9): Clear visual match, unambiguous.\n"
    "  - Medium (0.6-0.75): Good match but some visual uncertainty.\n"
    "  - Low (≤0.35): Ambiguous or no confident visual match.\n"
    "- **Justification:** Cite the specific image observation (frame ID, location in image, visual "
    "attributes) that drove the decision."
)


def _rank_frame_ids(
    frame_results: Dict,
    top_k: int,
    include_visual: bool = True,
    include_text: bool = False,
) -> List[str]:
    """Deduplicated, ranked list of frame chunk IDs from search results."""
    seen: set = set()
    ids: List[str] = []
    sources = []
    if include_visual:
        sources.append(frame_results.get("frame_visual", []))
    if include_text:
        sources.append(frame_results.get("text", []))
    for results in sources:
        for r in results:
            fid = r.chunk.id
            if fid not in seen:
                seen.add(fid)
                ids.append(fid)
            if len(ids) >= top_k:
                return ids
    return ids


def _load_frame_images(frame_chunks: List, max_images: int = 4) -> List:
    """Load frame images, stamping FRAME_ID in the top-left corner of each."""
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw, ImageFont as _ImageFont

    _FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]

    def _get_font(size: int = 22):
        for p in _FONT_PATHS:
            try:
                return _ImageFont.truetype(p, size)
            except Exception:
                pass
        return _ImageFont.load_default()

    images = []
    for chunk in frame_chunks[:max_images]:
        meta = chunk.metadata or {}
        path = meta.get("labeled_image_path") or meta.get("image_path")
        if not path or not os.path.isfile(path):
            continue
        try:
            img = _PILImage.open(path).convert("RGB")
            draw = _ImageDraw.Draw(img)
            label = f"FRAME_ID={chunk.id}"
            font = _get_font(22)
            # Measure text bounding box
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = len(label) * 13, 22
            pad = 5
            # Black background rectangle, then yellow text
            draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0))
            draw.text((pad, pad), label, fill=(255, 230, 0), font=font)
            images.append(img)
        except Exception as e:
            logger.debug("Could not load frame image {}: {}", path, e)
    return images


def _build_spatial_relations(target_vis, anchor_vis, obj_by_id) -> List[str]:
    lines = []
    for tr in target_vis[:3]:
        t_obj = obj_by_id.get(tr.chunk.id)
        if t_obj is None:
            continue
        t_pcd = getattr(t_obj, "pcd", None)
        if t_pcd is None or len(t_pcd.points) == 0:
            continue
        t_center = np.asarray(t_pcd.points).mean(axis=0)
        for ar in anchor_vis[:3]:
            a_obj = obj_by_id.get(ar.chunk.id)
            if a_obj is None:
                continue
            a_pcd = getattr(a_obj, "pcd", None)
            if a_pcd is None or len(a_pcd.points) == 0:
                continue
            a_center = np.asarray(a_pcd.points).mean(axis=0)
            dist = float(np.linalg.norm(t_center - a_center))
            lines.append(
                f"ID={tr.chunk.id} ({getattr(t_obj, 'label', '?')}) "
                f"is {dist:.2f}m from ID={ar.chunk.id} ({getattr(a_obj, 'label', '?')})"
            )
    return lines


def _run_grounding_query(
    scene_dir: str,
    query: str,
    top_k_objects: int = 10,
    top_k_frames: int = 4,
    max_frame_images: int = 4,
    retriever=None,
    objects=None,
) -> Dict[str, Any]:
    """RAG retrieval + LLM object selection — mirrors nr3d_eval._run_keysg_rag pipeline."""
    from pydantic import BaseModel, Field

    from keysg.rag.graph_context_retriever import GraphContextRetriever
    from keysg.rag.query_analysis import (
        _QuerySchema,
        SYSTEM_INSTRUCTIONS as _QUERY_ANALYSIS_INSTRUCTIONS,
    )
    from models.llm.openai_api import GPTInterface

    class ObjectSelection(BaseModel):
        object_id: Optional[str] = Field(
            default=None,
            description="Chosen object ID from candidates; null if none match",
        )
        reason: str = Field(description="Concise rationale for selection")
        confidence: float = Field(ge=0, le=1, description="Calibrated confidence 0-1")
        rejected_ids: List[str] = Field(
            default_factory=list, description="IDs considered but rejected"
        )
        guess_id: Optional[str] = Field(
            default=None, description="Closest guess if no confident selection"
        )

    gpt = GPTInterface()

    # Phase 1: query analysis
    anchor_objects: List[str] = []
    relation_polarity = None
    try:
        analysis = gpt.structured_prompt(
            f"User query: {query}",
            response_model=_QuerySchema,
            model="gpt-5-nano",
            instructions=_QUERY_ANALYSIS_INSTRUCTIONS,
        )
        target_q = analysis.target_object or query
        anchor_objects = analysis.anchor_objects or []
        relation_polarity = getattr(analysis, "relation_polarity", None)
    except Exception:
        target_q = query

    # Phase 2: RAG retrieval (use pre-built retriever if provided)
    if retriever is None:
        retriever = GraphContextRetriever(scene_dir)
        retriever.build_chunks()
        retriever.compute_embeddings(
            compute_frame_visual=True, compute_object_visual=True
        )
        retriever.build_faiss_index()

    target_results = retriever.search(
        target_q,
        top_k=top_k_objects,
        doc_types=["object"],
        object_modality="both",
    )
    target_vis = target_results.get("object_visual", [])

    anchor_vis: List = []
    if anchor_objects:
        anchor_query = " ".join(anchor_objects)
        anchor_results = retriever.search(
            anchor_query,
            top_k=top_k_objects,
            doc_types=["object"],
            object_modality="both",
        )
        anchor_vis = anchor_results.get("object_visual", [])

    # Frame retrieval (same as nr3d_eval)
    chunk_map = {c.id: c for c in retriever.chunks}
    frame_results = retriever.search(
        query,
        top_k=top_k_frames,
        doc_types=["frame"],
        object_modality="both",
        frame_modality="both",
    )
    top_frame_ids = _rank_frame_ids(
        frame_results, top_k_frames, include_visual=True, include_text=False
    )
    top_frame_chunks = [chunk_map[fid] for fid in top_frame_ids if fid in chunk_map]

    if not target_vis:
        return {
            "object_id": None,
            "label": None,
            "confidence": 0.0,
            "reason": "No candidates found",
        }

    # Phase 3: build context (same format as nr3d_eval)
    sections = [
        f"USER QUERY: {query}",
        f"PARSED TARGET: {target_q}",
        f"PARSED ANCHORS: {anchor_objects}",
    ]
    lines = ["Target Object Candidates:"]
    for i, r in enumerate(target_vis):
        lines.append(f"{i+1}. ID={r.chunk.id}). Desc={r.chunk.content}")
    sections.append("\n".join(lines))

    if anchor_vis:
        lines = ["Anchor Object Candidates:"]
        for i, r in enumerate(anchor_vis):
            lines.append(f"{i+1}. ID={r.chunk.id}. Desc={r.chunk.content}")
        sections.append("\n".join(lines))

    if top_frame_chunks:
        lines = ["Relevant Frames:"]
        for i, c in enumerate(top_frame_chunks):
            lines.append(f"{i+1}. FRAME_ID={c.id}. {c.content}")
        sections.append("\n".join(lines))

    if target_vis and anchor_vis and relation_polarity and objects:
        obj_by_id = {str(o.id): o for o in objects}
        spatial_lines = _build_spatial_relations(target_vis, anchor_vis, obj_by_id)
        if spatial_lines:
            sections.append(
                "Spatial Relations (target <-> anchor):\n" + "\n".join(spatial_lines)
            )

    context_text = "\n\n".join(sections)
    frame_images = _load_frame_images(top_frame_chunks, max_frame_images)

    # Phase 4: LLM selection
    try:
        sel = gpt.structured_prompt(
            context_text,
            response_model=ObjectSelection,
            model="gpt-5-mini",
            image=frame_images if frame_images else None,
            detail="high",
            instructions=_OBJECT_SELECTION_SYSTEM_PROMPT,
        )
        pred_id = sel.object_id or sel.guess_id
        confidence = sel.confidence
        reason = sel.reason
    except Exception as e:
        logger.warning("LLM object selection failed: {}", e)
        pred_id = target_vis[0].chunk.id
        confidence = float(target_vis[0].score)
        reason = "Fallback to top RAG hit"

    return {"object_id": pred_id, "confidence": confidence, "reason": reason}


_OPEN_QA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with access to a 3D scene description. "
    "Answer the user's question using only the provided scene context. "
    "Be specific: cite room IDs, object IDs, or frame IDs when relevant. "
    "If the context is insufficient to answer confidently, say so clearly."
)


def _run_open_qa(
    question: str,
    top_k_objects: int = 10,
    top_k_frames: int = 4,
    max_frame_images: int = 4,
    retriever=None,
    objects=None,
) -> Dict[str, Any]:
    """RAG retrieval + LLM for open-ended scene questions — same context pipeline as grounding."""
    from pydantic import BaseModel, Field

    from keysg.rag.query_analysis import (
        _QuerySchema,
        SYSTEM_INSTRUCTIONS as _QUERY_ANALYSIS_INSTRUCTIONS,
    )
    from models.llm.openai_api import GPTInterface

    class SceneAnswer(BaseModel):
        answer: str = Field(description="Direct answer to the question")
        reasoning: str = Field(
            description="Step-by-step reasoning citing scene evidence"
        )
        relevant_object_ids: List[str] = Field(
            default_factory=list, description="Object IDs mentioned in the answer"
        )

    gpt = GPTInterface()

    # Phase 1: query analysis
    anchor_objects: List[str] = []
    relation_polarity = None
    try:
        analysis = gpt.structured_prompt(
            f"User query: {question}",
            response_model=_QuerySchema,
            model="gpt-5-nano",
            instructions=_QUERY_ANALYSIS_INSTRUCTIONS,
        )
        target_q = analysis.target_object or question
        anchor_objects = analysis.anchor_objects or []
        relation_polarity = getattr(analysis, "relation_polarity", None)
    except Exception:
        target_q = question

    # Phase 2: RAG retrieval
    target_results = retriever.search(
        target_q,
        top_k=top_k_objects,
        doc_types=["object"],
        object_modality="both",
    )
    target_vis = target_results.get("object_visual", [])

    anchor_vis: List = []
    if anchor_objects:
        anchor_results = retriever.search(
            " ".join(anchor_objects),
            top_k=top_k_objects,
            doc_types=["object"],
            object_modality="both",
        )
        anchor_vis = anchor_results.get("object_visual", [])

    chunk_map = {c.id: c for c in retriever.chunks}
    frame_results = retriever.search(
        question,
        top_k=top_k_frames,
        doc_types=["frame"],
        object_modality="both",
        frame_modality="both",
    )
    top_frame_ids = _rank_frame_ids(frame_results, top_k_frames, include_visual=True)
    top_frame_chunks = [chunk_map[fid] for fid in top_frame_ids if fid in chunk_map]

    # Phase 3: build context (same format as grounding)
    sections = [
        f"USER QUERY: {question}",
        f"PARSED TARGET: {target_q}",
        f"PARSED ANCHORS: {anchor_objects}",
    ]
    if target_vis:
        lines = ["Relevant Object Candidates:"]
        for i, r in enumerate(target_vis):
            lines.append(f"{i+1}. ID={r.chunk.id}. Desc={r.chunk.content}")
        sections.append("\n".join(lines))

    if anchor_vis:
        lines = ["Anchor Object Candidates:"]
        for i, r in enumerate(anchor_vis):
            lines.append(f"{i+1}. ID={r.chunk.id}. Desc={r.chunk.content}")
        sections.append("\n".join(lines))

    if top_frame_chunks:
        lines = ["Relevant Frames:"]
        for i, c in enumerate(top_frame_chunks):
            lines.append(f"{i+1}. FRAME_ID={c.id}. {c.content}")
        sections.append("\n".join(lines))

    if target_vis and anchor_vis and relation_polarity and objects:
        obj_by_id = {str(o.id): o for o in objects}
        spatial_lines = _build_spatial_relations(target_vis, anchor_vis, obj_by_id)
        if spatial_lines:
            sections.append("Spatial Relations:\n" + "\n".join(spatial_lines))

    context_text = "\n\n".join(sections)
    frame_images = _load_frame_images(top_frame_chunks, max_frame_images)

    # Phase 4: LLM answer
    try:
        resp = gpt.structured_prompt(
            context_text,
            response_model=SceneAnswer,
            model="gpt-5-mini",
            image=frame_images if frame_images else None,
            detail="high",
            instructions=_OPEN_QA_SYSTEM_PROMPT,
        )
        return resp.model_dump()
    except Exception as e:
        logger.warning("Open-ended QA failed: {}", e)
        return {"answer": str(e), "reasoning": "", "relevant_object_ids": []}


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------


class KeySGVisualizer:
    """Interactive Viser visualizer for KeySG scene graphs."""

    def __init__(self, scene_dir: str, port: int = 8080):
        self.scene_dir = scene_dir
        self.port = port
        self.server: Optional[viser.ViserServer] = None

        # Scene data
        self.floors: Dict = {}
        self.rooms: Dict = {}
        self.objects: List = []

        # Viser handles keyed by object ID
        self._obj_handles: Dict[str, Any] = {}
        self._obj_colors: Dict[str, np.ndarray] = {}  # per-instance palette color
        self._obj_pts: Dict[str, np.ndarray] = {}  # point positions for bbox
        self._floor_handles: Dict[str, Any] = {}
        self._room_handles: Dict[str, Any] = {}
        self._frustum_handles: Dict[str, Any] = {}
        self._bbox_handle: Optional[Any] = None

        # State
        self._color_mode: str = "instance"  # "instance" | "rgb"
        self._flip_z: bool = False
        self._grounding_retriever = None  # cached after first query

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info("Loading KeySG scene from {}", self.scene_dir)
        self.floors = get_floors(self.scene_dir)
        self.rooms = get_rooms(self.scene_dir)
        nodes_nested = load_scene_nodes(self.scene_dir)
        self.objects = get_objects(nodes_nested)
        logger.info(
            "Loaded {} floors, {} rooms, {} objects",
            len(self.floors),
            len(self.rooms),
            len(self.objects),
        )

    # ------------------------------------------------------------------
    # Point-cloud helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcd_to_arrays(pcd):
        if pcd is None:
            return None, None
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return None, None
        cols = (
            (np.asarray(pcd.colors) * 255).astype(np.uint8)
            if pcd.has_colors()
            else np.full((len(pts), 3), 128, dtype=np.uint8)
        )
        return pts.astype(np.float32), cols

    def _transform_pts(self, pts: np.ndarray) -> np.ndarray:
        if self._flip_z:
            pts = pts.copy()
            pts[:, 2] *= -1
        return pts

    def _rebuild_scene(self) -> None:
        """Remove and re-add all scene layers (called on flip / color-mode change)."""
        for handles in (
            self._floor_handles,
            self._room_handles,
            self._obj_handles,
            self._frustum_handles,
        ):
            for h in handles.values():
                try:
                    h.remove()
                except Exception:
                    pass
            handles.clear()
        self._clear_bbox()
        self._add_floors()
        self._add_rooms()
        self._add_objects()
        self._add_keyframes()

    # ------------------------------------------------------------------
    # Add layers
    # ------------------------------------------------------------------

    def _add_floors(self) -> None:
        palette = _palette(max(len(self.floors), 1))
        for i, (fid, floor) in enumerate(self.floors.items()):
            pcd = getattr(floor, "pcd", None)
            if pcd is None:
                continue
            pts, _ = self._pcd_to_arrays(pcd)
            if pts is None:
                continue
            pts = self._transform_pts(pts)
            color = np.tile(palette[i % len(palette)], (len(pts), 1))
            handle = self.server.scene.add_point_cloud(
                f"/floors/{fid}",
                points=pts,
                colors=color,
                point_size=0.02,
            )
            handle.visible = False  # off by default
            self._floor_handles[fid] = handle
            centroid = pts.mean(axis=0)
            self.server.scene.add_label(
                f"/floors/{fid}/label",
                text=f"Floor {fid}",
                position=centroid + np.array([0, 0, 0.5], dtype=np.float32),
            )

    def _add_rooms(self) -> None:
        palette = _palette(max(len(self.rooms), 1))
        for i, (rid, room) in enumerate(self.rooms.items()):
            pcd = getattr(room, "pcd", None)
            if pcd is None:
                continue
            pts, _ = self._pcd_to_arrays(pcd)
            if pts is None:
                continue
            pts = self._transform_pts(pts)
            color = np.tile(palette[i % len(palette)], (len(pts), 1))
            handle = self.server.scene.add_point_cloud(
                f"/rooms/{rid}",
                points=pts,
                colors=color,
                point_size=0.015,
            )
            handle.visible = False  # off by default
            self._room_handles[rid] = handle

    def _add_objects(self, color_mode: Optional[str] = None) -> None:
        _EXCLUDE = {"wall", "floor", "ceiling"}
        palette = _palette(max(len(self.objects), 1))
        mode = color_mode if color_mode is not None else self._color_mode

        for i, obj in enumerate(self.objects):
            label = (getattr(obj, "label", "") or "").lower()
            if any(e in label for e in _EXCLUDE):
                continue
            pcd = getattr(obj, "pcd", None)
            if pcd is None or len(pcd.points) < 20:
                continue
            pts, rgb_colors = self._pcd_to_arrays(pcd)
            if pts is None:
                continue
            pts = self._transform_pts(pts)

            obj_id = str(getattr(obj, "id", f"obj_{i}"))
            instance_color = palette[i % len(palette)]
            self._obj_colors[obj_id] = instance_color
            self._obj_pts[obj_id] = pts

            if mode == "rgb":
                display_color = rgb_colors
            else:
                display_color = np.tile(instance_color, (len(pts), 1))

            # Remove old handle if present
            old = self._obj_handles.pop(obj_id, None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass

            handle = self.server.scene.add_point_cloud(
                f"/objects/{obj_id}",
                points=pts,
                colors=display_color,
                point_size=0.008,
            )
            self._obj_handles[obj_id] = handle

    def _add_keyframes(self) -> None:
        """Add camera frustums at their world-space positions with RGB thumbnails."""
        seg_dir = os.path.join(self.scene_dir, "segmentation")
        if not os.path.isdir(seg_dir):
            return

        for floor_name in os.listdir(seg_dir):
            floor_path = os.path.join(seg_dir, floor_name)
            if not os.path.isdir(floor_path) or not floor_name.startswith("floor_"):
                continue
            for room_name in os.listdir(floor_path):
                room_path = os.path.join(floor_path, room_name)
                if not os.path.isdir(room_path) or not room_name.startswith("room_"):
                    continue
                poses_file = os.path.join(room_path, "keyframe_poses.json")
                if not os.path.isfile(poses_file):
                    continue

                with open(poses_file) as f:
                    poses: Dict[str, List] = json.load(f)

                kf_img_dir = os.path.join(room_path, "keyframes")

                for idx_str, pose_list in poses.items():
                    pose = np.array(pose_list, dtype=np.float64)
                    if self._flip_z:
                        S = np.diag([1.0, 1.0, -1.0])
                        pose[:3, :3] = S @ pose[:3, :3] @ S
                        pose[:3, 3] = S @ pose[:3, 3]
                    wxyz, pos = _pose_to_wxyz_pos(pose)

                    img_path = os.path.join(kf_img_dir, f"frame_{int(idx_str):06d}.jpg")
                    if not os.path.isfile(img_path):
                        img_path = os.path.join(
                            kf_img_dir, f"frame_{int(idx_str):06d}.png"
                        )
                    img = _load_thumbnail(img_path)

                    name = f"/keyframes/{floor_name}/{room_name}/{idx_str}"
                    try:
                        handle = self.server.scene.add_camera_frustum(
                            name=name,
                            fov=np.deg2rad(60.0),
                            aspect=4.0 / 3.0,
                            scale=0.15,
                            wxyz=wxyz,
                            position=pos,
                            image=img,
                            color=(180, 220, 255),
                        )
                        self._frustum_handles[name] = handle
                    except Exception as e:
                        logger.debug("Could not add frustum {}: {}", name, e)

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def _draw_bbox(self, obj_id: str) -> None:
        """Draw a red wireframe AABB around the given object."""
        # Remove previous bbox
        if self._bbox_handle is not None:
            try:
                self._bbox_handle.remove()
            except Exception:
                pass
            self._bbox_handle = None

        pts = self._obj_pts.get(obj_id)
        if pts is None or len(pts) == 0:
            return

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical edges
        ]
        seg_pts = np.array(
            [[corners[a], corners[b]] for a, b in edges], dtype=np.float32
        )  # (12, 2, 3)

        try:
            self._bbox_handle = self.server.scene.add_line_segments(
                "/grounding/bbox",
                points=seg_pts,
                colors=np.array([220, 30, 30], dtype=np.uint8),
                line_width=5.0,
            )
        except Exception as e:
            logger.warning("Could not draw bbox: {}", e)

    def _clear_bbox(self) -> None:
        if self._bbox_handle is not None:
            try:
                self._bbox_handle.remove()
            except Exception:
                pass
            self._bbox_handle = None

    # ------------------------------------------------------------------
    # Retriever (cached after first use)
    # ------------------------------------------------------------------

    def _ensure_retriever(self):
        """Lazily build and cache the GraphContextRetriever (keeps CLIP model loaded)."""
        if self._grounding_retriever is None:
            from keysg.rag.graph_context_retriever import GraphContextRetriever

            logger.info(
                "Building grounding retriever (first query — CLIP model loading)…"
            )
            r = GraphContextRetriever(self.scene_dir)
            r.build_chunks()
            r.compute_embeddings(compute_frame_visual=True, compute_object_visual=True)
            r.build_faiss_index()
            self._grounding_retriever = r
            logger.info("Retriever ready — subsequent queries will reuse cached model.")
        return self._grounding_retriever

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        # -- Layer toggles --
        with self.server.gui.add_folder("Layers"):
            chk_floors = self.server.gui.add_checkbox("Floors", initial_value=False)
            chk_rooms = self.server.gui.add_checkbox("Rooms", initial_value=False)
            chk_objects = self.server.gui.add_checkbox("Objects", initial_value=True)
            chk_kf = self.server.gui.add_checkbox("Keyframes", initial_value=True)
            chk_flip_z = self.server.gui.add_checkbox("Flip Z", initial_value=False)

            @chk_floors.on_update
            def _(_):
                for h in self._floor_handles.values():
                    h.visible = chk_floors.value

            @chk_rooms.on_update
            def _(_):
                for h in self._room_handles.values():
                    h.visible = chk_rooms.value

            @chk_objects.on_update
            def _(_):
                for h in self._obj_handles.values():
                    h.visible = chk_objects.value

            @chk_kf.on_update
            def _(_):
                for h in self._frustum_handles.values():
                    h.visible = chk_kf.value

            @chk_flip_z.on_update
            def _(_):
                self._flip_z = chk_flip_z.value
                self._rebuild_scene()

        # -- Object color mode --
        with self.server.gui.add_folder("Object Coloring"):
            color_mode_dd = self.server.gui.add_dropdown(
                "Color Mode",
                options=["Instance Segments", "RGB"],
                initial_value="Instance Segments",
            )

            @color_mode_dd.on_update
            def _(_):
                self._color_mode = "rgb" if color_mode_dd.value == "RGB" else "instance"
                self._add_objects(color_mode=self._color_mode)

        # -- Manual bbox by object ID --
        with self.server.gui.add_folder("Draw BBox by ID"):
            bbox_id_input = self.server.gui.add_text("Object ID", initial_value="")
            bbox_id_btn = self.server.gui.add_button("Draw BBox")
            bbox_id_result = self.server.gui.add_markdown(
                "_Enter an object ID and click Draw BBox._"
            )

            @bbox_id_btn.on_click
            def _(_):
                oid = bbox_id_input.value.strip()
                if not oid:
                    return
                self._clear_bbox()
                if oid in self._obj_pts:
                    obj_label = next(
                        (
                            getattr(o, "label", oid)
                            for o in self.objects
                            if str(getattr(o, "id", "")) == oid
                        ),
                        oid,
                    )
                    self._draw_bbox(oid)
                    _pts = self._obj_pts[oid]
                    _center = (_pts.min(axis=0) + _pts.max(axis=0)) / 2
                    bbox_id_result.content = (
                        f"**Drawing bbox for:** {obj_label} (ID: `{oid}`)\n\n"
                        f"**BBox Center:** x={_center[0]:.3f}, y={_center[1]:.3f}, z={_center[2]:.3f}"
                    )
                else:
                    bbox_id_result.content = f"_Object ID `{oid}` not found in scene._"

        # -- Object grounding --
        with self.server.gui.add_folder("Object Grounding"):
            grounding_input = self.server.gui.add_text(
                "Query", initial_value="", multiline=True
            )
            grounding_btn = self.server.gui.add_button("Find Object")
            grounding_result = self.server.gui.add_markdown(
                "_Enter a query and click Find._"
            )

            @grounding_btn.on_click
            def _(_):
                q = grounding_input.value.strip()
                if not q:
                    return
                grounding_result.content = "_Searching…_"
                self._clear_bbox()
                try:
                    result = _run_grounding_query(
                        self.scene_dir,
                        q,
                        retriever=self._ensure_retriever(),
                        objects=self.objects,
                    )
                    obj_id = result.get("object_id")
                    confidence = result.get("confidence", 0.0)
                    reason = result.get("reason", "")
                    if obj_id:
                        obj_label = next(
                            (
                                getattr(o, "label", obj_id)
                                for o in self.objects
                                if str(getattr(o, "id", "")) == str(obj_id)
                            ),
                            obj_id,
                        )
                        self._draw_bbox(str(obj_id))
                        _pts = self._obj_pts.get(str(obj_id))
                        _center_str = ""
                        if _pts is not None and len(_pts) > 0:
                            _center = (_pts.min(axis=0) + _pts.max(axis=0)) / 2
                            _center_str = f"\n\n**BBox Center:** x={_center[0]:.3f}, y={_center[1]:.3f}, z={_center[2]:.3f}"
                        grounding_result.content = (
                            f"**Found:** {obj_label} (ID: `{obj_id}`)\n\n"
                            f"**Confidence:** {confidence:.2f}\n\n"
                            f"**Reasoning:** {reason}"
                            f"{_center_str}"
                        )
                    else:
                        grounding_result.content = f"_No match found._\n\n{reason}"
                except Exception as e:
                    logger.error("Grounding query failed: {}", e)
                    grounding_result.content = f"_Error: {e}_"

        # -- Open-ended Q&A --
        with self.server.gui.add_folder("Open-Ended Q&A"):
            qa_input = self.server.gui.add_text(
                "Question", initial_value="", multiline=True
            )
            qa_btn = self.server.gui.add_button("Ask")
            qa_result = self.server.gui.add_markdown("_Ask anything about this scene._")

            @qa_btn.on_click
            def _(_):
                q = qa_input.value.strip()
                if not q:
                    return
                qa_result.content = "_Thinking…_"
                try:
                    response = _run_open_qa(
                        q,
                        retriever=self._ensure_retriever(),
                        objects=self.objects,
                    )
                    answer = response.get("answer", "")
                    reasoning = response.get("reasoning", "")
                    qa_result.content = (
                        f"**Answer:** {answer}\n\n**Reasoning:** {reasoning}"
                    )
                except Exception as e:
                    logger.error("Open-ended query failed: {}", e)
                    qa_result.content = f"_Error: {e}_"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Load scene, start Viser server, and block."""
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        actual_port = self.server.get_port()
        logger.info("Viser server started at http://localhost:{}", actual_port)
        logger.info("Scene dir: {}", self.scene_dir)

        self._load()
        self.server.scene.reset()
        self.server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.01)

        self._add_floors()
        self._add_rooms()
        self._add_objects()
        self._add_keyframes()
        self._build_gui()

        logger.info("Scene loaded — open http://localhost:{} and refresh if switching scenes.", actual_port)
        logger.info("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="KeySG interactive scene visualizer")
    parser.add_argument(
        "--scene_dir", "-s", required=True, help="Path to pipeline output directory"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080, help="Viser server port (default: 8080)"
    )
    args = parser.parse_args()
    KeySGVisualizer(args.scene_dir, port=args.port).run()


if __name__ == "__main__":
    main()
