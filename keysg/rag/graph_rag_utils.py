"""Utility helpers for Graph RAG retrieval.

This module centralizes auxiliary logic used by `GraphContextRetriever` to keep
its main implementation concise:

Functions provided:
    ensure_clip(retriever, clip_config)
    gather_frame_chunks(chunks)
    compute_frame_visual_embeddings(retriever, clip_config=None, use_cache=True)
    build_frame_visual_faiss_index(retriever, use_cache=True)

The functions operate directly on a `GraphContextRetriever` instance to mutate
its state (lazy initialization, caching, FAISS index building). This avoids
duplicating large code blocks inside the main retriever class while preserving
the public method signatures for backwards compatibility.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Sequence, Set
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover

    def _tqdm(it, **kwargs):  # type: ignore[misc]
        return it


try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

# Optional CLIP feature extractor
try:  # pragma: no cover - optional dependency
    from keysg.utils.clip_utils import (
        CLIPFeatureExtractor,
        DEFAULT_CLIP_CONFIG as _DEFAULT_CLIP_CONFIG,
    )
except ImportError:  # pragma: no cover
    # Fallback for when running from within the package
    try:
        import sys
        import os as _os

        _project_root = _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", "..")
        )
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from keysg.utils.clip_utils import (
            CLIPFeatureExtractor,
            DEFAULT_CLIP_CONFIG as _DEFAULT_CLIP_CONFIG,
        )
    except Exception:
        CLIPFeatureExtractor = None  # type: ignore
        _DEFAULT_CLIP_CONFIG = None  # type: ignore


@dataclass
class Chunk:
    """A single semantic unit to embed and index.

    Attributes:
        id: Stable unique identifier (e.g. floor_0, room_0_2, frame_0_2_1299, object_0_2_1299_obj_ab12)
        doc_type: One of {floor, room, frame, object}
        content: Rendered natural language text used for embedding
        metadata: Arbitrary metadata fields (floor_id, room_id, frame_index, object_id, path ...)
        embedding: Optional vector (filled post-encoding)
    """

    id: str
    doc_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None  # numpy array after encoding

    def to_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "doc_type": self.doc_type,
            "content": self.content,
            "metadata": self.metadata,
        }


def synthesize_object_text(obj: Dict[str, Any]) -> str:
    """Create a canonical descriptive string for an object entry.

    Expected keys (best-effort): name, description, affordances(list), state, location description
    """
    name = obj.get("name") or obj.get("id") or "object"
    desc = obj.get("description") or ""
    affordances = obj.get("affordances") or []
    attributes = obj.get("attributes") or []
    if isinstance(attributes, str):
        attributes = [attributes]
    if attributes:
        desc += (
            " It has attributes: "
            + ", ".join(a for a in attributes if isinstance(a, str))
            + "."
        )
    spatial_relations = obj.get("spatial_relations") or []
    if isinstance(spatial_relations, list) and spatial_relations:
        desc += (
            " Spatial relations: "
            + "; ".join(r for r in spatial_relations if isinstance(r, str))
            + "."
        )
    if isinstance(affordances, str):
        affordances_list = [affordances]
    else:
        affordances_list = [a for a in affordances if isinstance(a, str)]
    afford_txt = (
        ", ".join(affordances_list) if affordances_list else "(no affordances listed)"
    )
    state = obj.get("state") or obj.get("status") or "unknown state"
    loc = (
        obj.get("location description")
        or obj.get("location")
        or "an unspecified location"
    )

    return (
        f"Name: {name}. Description: {desc} "
        f"It can be used to {afford_txt}. It is currently {state} and is located {loc}."
    )


def ensure_text(*parts: Any) -> str:
    """Join arbitrary possibly None parts into a clean single string."""
    out = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, (list, tuple)):
            out.append("; ".join(str(x) for x in p if x))
        else:
            out.append(str(p))
    return "\n".join(out)


def ensure_clip(retriever, clip_config: Optional[Dict[str, Any]] = None):
    """Lazy-initialize CLIP extractor on the retriever.

    Raises:
        RuntimeError: if CLIP dependencies are missing.
    """
    if CLIPFeatureExtractor is None:
        raise RuntimeError(
            "CLIPFeatureExtractor unavailable. Install open_clip and ensure utils.clip_utils imported."
        )
    if retriever.clip is None:
        cfg = (_DEFAULT_CLIP_CONFIG or {}).copy()
        if clip_config:
            cfg.update(clip_config)
        retriever.clip = CLIPFeatureExtractor(cfg)
        retriever.clip_model_id = cfg.get("model_name")


def gather_frame_chunks(chunks) -> List[Tuple[int, Any]]:
    """Return list of (index, chunk) pairs for frame chunks."""
    return [
        (idx, c)
        for idx, c in enumerate(chunks)
        if getattr(c, "doc_type", None) == "frame"
    ]


def gather_object_chunks(chunks) -> List[Tuple[int, Any]]:
    """Return list of (index, chunk) pairs for object chunks."""
    return [
        (idx, c)
        for idx, c in enumerate(chunks)
        if getattr(c, "doc_type", None) == "object"
    ]


def compute_frame_visual_embeddings(
    retriever,
    clip_config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """Compute (or load cached) CLIP image embeddings for frame chunks.

    Stores results in retriever.frame_visual_embeddings & related metadata.
    Returns ndarray or None (if no frames or failures).
    """
    if not retriever.chunks:
        retriever.build_chunks()
    frames = gather_frame_chunks(retriever.chunks)
    if not frames:
        return None

    # Cache attempt
    if use_cache:
        emb_exists = os.path.exists(retriever.frame_vis_emb_path)
        meta_exists = os.path.exists(retriever.frame_vis_meta_path)
        if emb_exists and meta_exists:
            try:
                with open(retriever.frame_vis_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                target_model = (clip_config or {}).get(
                    "model_name"
                ) or retriever.clip_model_id
                model_ok = (
                    meta.get("clip_model_id") == target_model
                ) or target_model is None
                # Check total chunk count to detect chunk-list rebuilds
                chunks_ok = meta.get("total_chunks", -1) == len(retriever.chunks)
                emb = np.load(retriever.frame_vis_emb_path)
                frame_indices = meta.get("frame_chunk_indices", [])
                shape_ok = bool(frame_indices) and emb.shape[0] == len(frame_indices)
                # Verify every cached position still points to a frame chunk
                indices_ok = chunks_ok and all(
                    i < len(retriever.chunks)
                    and retriever.chunks[i].doc_type == "frame"
                    for i in frame_indices
                )
                if model_ok and shape_ok and indices_ok:
                    retriever.frame_visual_embeddings = emb
                    retriever.frame_visual_chunk_indices = frame_indices
                    retriever.clip_model_id = meta.get("clip_model_id")
                    logger.info(
                        "Loaded cached frame visual embeddings: shape {} (model={})",
                        emb.shape,
                        retriever.clip_model_id,
                    )
                    return emb
                else:
                    logger.info(
                        "Discarding cached frame visuals "
                        "(model_ok=%s shape_ok=%s chunks_ok=%s indices_ok=%s)",
                        model_ok,
                        shape_ok,
                        chunks_ok,
                        indices_ok,
                    )
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to load cached frame visual embeddings: {}", e)

    # Compute fresh
    ensure_clip(retriever, clip_config)
    img_feats_list = []
    frame_indices: List[int] = []
    missing_paths = 0
    for chunk_idx, frame_chunk in _tqdm(
        frames, desc="CLIP-encoding frames", unit="frame"
    ):
        img_path = (
            frame_chunk.metadata.get("image_path") if frame_chunk.metadata else None
        )
        if not img_path or not os.path.exists(img_path):
            missing_paths += 1
            continue
        try:
            from PIL import Image  # local imported only when needed

            with Image.open(img_path) as im:
                feat = retriever.clip.get_img_feats(im)  # type: ignore[attr-defined]
            img_feats_list.append(feat)
            frame_indices.append(chunk_idx)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed image embedding for {}: {}", img_path, e)

    if not img_feats_list:
        logger.warning(
            "No frame visual embeddings computed (missing images: {})", missing_paths
        )
        return None

    retriever.frame_visual_embeddings = np.vstack(img_feats_list).astype("float32")
    retriever.frame_visual_chunk_indices = frame_indices

    # Cache save
    np.save(retriever.frame_vis_emb_path, retriever.frame_visual_embeddings)
    with open(retriever.frame_vis_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "created": time.time(),
                "n_frames": len(frame_indices),
                "clip_model_id": retriever.clip_model_id,
                "frame_chunk_indices": frame_indices,
                "total_chunks": len(retriever.chunks),
            },
            f,
            indent=2,
        )
    logger.info(
        "Computed frame visual embeddings ({} frames, dim={})",
        *retriever.frame_visual_embeddings.shape,
    )
    return retriever.frame_visual_embeddings


def build_frame_visual_faiss_index(retriever, use_cache: bool = True):
    if faiss is None:
        raise RuntimeError(
            "faiss library not installed. Required for visual frame index."
        )
    if (
        use_cache
        and os.path.exists(retriever.frame_vis_index_path)
        and os.path.exists(retriever.frame_vis_meta_path)
    ):
        try:
            # Ensure embeddings also loaded
            if retriever.frame_visual_embeddings is None:
                if os.path.exists(retriever.frame_vis_emb_path):
                    retriever.frame_visual_embeddings = np.load(
                        retriever.frame_vis_emb_path
                    )
            retriever.frame_visual_index = faiss.read_index(
                retriever.frame_vis_index_path
            )
            with open(retriever.frame_vis_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            retriever.frame_visual_chunk_indices = meta.get("frame_chunk_indices", [])
            # Validate counts
            if (
                retriever.frame_visual_embeddings is not None
                and retriever.frame_visual_index.ntotal
                == retriever.frame_visual_embeddings.shape[0]
            ):
                logger.info(
                    "Loaded cached frame visual FAISS index ({} vectors)",
                    retriever.frame_visual_index.ntotal,
                )
                return
            else:
                logger.info(
                    "Cached frame visual index mismatch (index ntotal=%s, emb shape=%s) - rebuilding",
                    getattr(retriever.frame_visual_index, "ntotal", "NA"),
                    (
                        None
                        if retriever.frame_visual_embeddings is None
                        else retriever.frame_visual_embeddings.shape
                    ),
                )
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load cached frame visual index: {}", e)

    if retriever.frame_visual_embeddings is None:
        compute_frame_visual_embeddings(retriever)
    if retriever.frame_visual_embeddings is None:
        return
    d = retriever.frame_visual_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(retriever.frame_visual_embeddings)
    faiss.write_index(index, retriever.frame_vis_index_path)
    retriever.frame_visual_index = index
    logger.info(
        "Built frame visual FAISS index ({} vectors, dim={})",
        index.ntotal,
        d,
    )


def load_scene_description(output_dir: str):
    """Load floor summaries + room VLM files.

    Returns: (floor_summaries dict, room_vlm dict, room_index_paths dict)
    Missing files tolerated.
    """
    floor_path = os.path.join(output_dir, "floor_summaries.json")
    scene_index_path = os.path.join(output_dir, "scene_description_index.json")
    floor_summaries: Dict[str, Any] = {}
    room_vlm: Dict[str, Any] = {}
    room_index_paths: Dict[str, str] = {}

    if os.path.exists(floor_path):
        try:
            with open(floor_path, "r", encoding="utf-8") as f:
                floor_summaries = json.load(f)
        except Exception as e:
            logger.warning("Failed reading floor_summaries.json: {}", e)

    if os.path.exists(scene_index_path):
        try:
            with open(scene_index_path, "r", encoding="utf-8") as f:
                scene_index = json.load(f)
            for r in scene_index.get("rooms", []):
                p = r.get("vlm_path")
                rid = r.get("id")
                if not p or not rid:
                    continue
                # Resolve relative paths: try several strategies.
                if not os.path.isabs(p) and not os.path.exists(p):
                    resolved = False
                    # Strategy 1: walk up from output_dir to find correct base
                    base = output_dir
                    for _ in range(10):
                        base = os.path.dirname(base)
                        if not base or base == os.path.dirname(base):
                            break
                        candidate = os.path.join(base, p)
                        if os.path.exists(candidate):
                            p = candidate
                            resolved = True
                            break
                    # Strategy 2: the vlm_path may reference a different pipeline
                    # run (e.g. keysg_rag_dense vs keysg_rag_key) but the
                    # segmentation/ subtree is identical.  Try resolving the
                    # segmentation-relative suffix under the current output_dir.
                    if not resolved:
                        seg_marker = os.sep + "segmentation" + os.sep
                        seg_idx = p.find(seg_marker)
                        if seg_idx >= 0:
                            suffix = p[seg_idx + 1 :]  # "segmentation/..."
                            candidate = os.path.join(output_dir, suffix)
                            if os.path.exists(candidate):
                                p = candidate
                if not os.path.exists(p):
                    continue
                try:
                    with open(p, "r", encoding="utf-8") as rf:
                        room_vlm[rid] = json.load(rf)
                        room_index_paths[rid] = p
                except Exception as e:  # pragma: no cover
                    logger.warning("Failed loading room VLM {}: {}", p, e)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed reading scene_description_index.json: {}", e)
    else:
        # Fallback: recursive search (legacy layout)
        seg_dir = os.path.join(output_dir, "segmentation")
        if os.path.isdir(seg_dir):
            for root, _dirs, files in os.walk(seg_dir):
                for fname in files:
                    if fname.endswith("_vlm.json"):
                        p = os.path.join(root, fname)
                        try:
                            with open(p, "r", encoding="utf-8") as rf:
                                data = json.load(rf)
                                rid = data.get("id")
                                if rid:
                                    room_vlm[rid] = data
                                    room_index_paths[rid] = p
                        except Exception:  # pragma: no cover
                            continue

    logger.info(
        "Loaded scene description: floors={} rooms={}",
        len(floor_summaries),
        len(room_vlm),
    )
    return floor_summaries, room_vlm, room_index_paths


def build_chunks_from_descriptions(
    floor_summaries: Dict[str, Any],
    room_vlm: Dict[str, Any],
    room_index_paths: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
) -> List[Chunk]:
    """Construct chunk list from description dicts (floors, rooms, frames, objects)."""
    chunks: List[Chunk] = []

    # Floors
    for floor_id, payload in (floor_summaries or {}).items():
        cap = payload.get("floor_caption") or payload.get("caption") or ""
        rooms_list = payload.get("rooms") or []
        room_summaries = []
        for r in rooms_list:
            rc = r.get("room_caption") or r.get("caption") or ""
            rid = r.get("room_id") or r.get("id")
            if rid:
                room_summaries.append(f"Room {rid}: {rc}")
        content = ensure_text(
            f"Floor {floor_id}",
            cap,
            "Rooms:",
            room_summaries,
        )
        chunks.append(
            Chunk(
                id=f"floor_{floor_id}",
                doc_type="floor",
                content=content,
                metadata={"floor_id": floor_id},
            )
        )

    # Rooms + frames + objects
    room_items = list((room_vlm or {}).items())
    for rid, room_data in _tqdm(
        room_items, desc="Building chunks", unit="room", leave=False
    ):
        summary = room_data.get("summary")
        if isinstance(summary, dict):
            summary_txt = (
                summary.get("room_type", "")
                + " "
                + summary.get("room_summary", "")
                + " "
                + summary.get("layout", "")
            )
        else:
            summary_txt = summary or ""
        chunks.append(
            Chunk(
                id=f"room_{rid}",
                doc_type="room",
                content=ensure_text(f"Room {rid}", summary_txt),
                metadata={"room_id": rid, "floor_id": rid.split("_")[0]},
            )
        )

        # Derive labeled_keyframes directory from the room VLM path
        room_dir = ""
        if room_index_paths and rid in room_index_paths:
            room_dir = os.path.dirname(room_index_paths[rid])

        frames = room_data.get("frames") or []
        for fr in _tqdm(frames, desc=f"  room {rid} frames", unit="frame", leave=False):
            idx = fr.get("index")
            if idx is None:
                continue
            desc = fr.get("description") or {}
            cap = desc.get("caption") or ""
            room_type = desc.get("room_type_guess", "")
            scene_layout = desc.get("scene_layout", "")
            detailed_desc = desc.get("description", "")
            node_tags = desc.get("objects") or []
            frame_id = f"frame_{rid}_{idx}"
            content = ensure_text(
                f"Frame {idx} (room {rid})",
                f"Caption: {cap}",
                f"Room Type: {room_type}",
                f"Scene Layout: {scene_layout}",
                f"Description: {detailed_desc}",
                f"Node Tags: {node_tags}",
            )
            labeled_path = ""
            if room_dir:
                candidate = os.path.join(
                    room_dir, "labeled_keyframes", f"frame_{idx:06d}.png"
                )
                if os.path.isfile(candidate):
                    labeled_path = candidate
            chunks.append(
                Chunk(
                    id=frame_id,
                    doc_type="frame",
                    content=content,
                    metadata={
                        "room_id": rid,
                        "frame_index": idx,
                        "image_path": fr.get("path"),
                        "labeled_image_path": labeled_path,
                        "node_tags": node_tags,
                    },
                )
            )

            objs = []
            if isinstance(desc, dict):
                objs = desc.get("objects") or []
            for obj in objs:
                oid = obj.get("id") or obj.get("name")
                if not oid:
                    continue
                chunks.append(
                    Chunk(
                        id=oid,
                        doc_type="object",
                        content=synthesize_object_text(obj),
                        metadata={
                            "room_id": rid,
                            "frame_index": idx,
                            "object_id": oid,
                            "name": obj.get("name") or oid,
                        },
                    )
                )

    # Deduplicate object chunks (same object in multiple frames)
    seen_obj_ids: Set[str] = set()
    filtered_chunks = []
    for chunk in chunks:
        if chunk.doc_type == "object":
            if chunk.id in seen_obj_ids:
                continue
            seen_obj_ids.add(chunk.id)
        filtered_chunks.append(chunk)
    chunks = filtered_chunks

    # Second pass: node-pickle objects (carry vlm_description + stable IDs).
    # Replaces the room-VLM chunk for the same object_id with a richer version.
    if output_dir:
        import pickle

        seg_dir = os.path.join(output_dir, "segmentation")
        if os.path.isdir(seg_dir):
            # Pre-collect all pkl paths for accurate tqdm total
            pkl_entries: List[Tuple[str, str]] = []  # (full_path, room_id)
            for floor_d in sorted(os.listdir(seg_dir)):
                floor_path = os.path.join(seg_dir, floor_d)
                if not os.path.isdir(floor_path):
                    continue
                for room_d in sorted(os.listdir(floor_path)):
                    nodes_dir = os.path.join(floor_path, room_d, "nodes")
                    if not os.path.isdir(nodes_dir):
                        continue
                    room_id = room_d.replace("room_", "")
                    for fname in sorted(os.listdir(nodes_dir)):
                        if fname.endswith(".pkl"):
                            pkl_entries.append(
                                (os.path.join(nodes_dir, fname), room_id)
                            )
            for pkl_path, room_id in _tqdm(
                pkl_entries, desc="Indexing node pickles", unit="pkl", leave=False
            ):
                try:
                    with open(pkl_path, "rb") as f:
                        nd = pickle.load(f)
                except Exception:
                    continue
                if not isinstance(nd, dict):
                    continue
                obj_id = nd.get("id")
                if not obj_id:
                    continue
                vlm = nd.get("vlm_description") or {}
                label = nd.get("label") or "object"
                obj_for_text = {
                    "name": vlm.get("name") or label,
                    "description": vlm.get("description") or "",
                    "attributes": vlm.get("attributes") or [],
                    "affordances": vlm.get("affordances") or [],
                    "state": vlm.get("state") or "",
                    "location description": vlm.get("location description") or "",
                    "spatial_relations": vlm.get("spatial_relations") or [],
                }
                new_chunk = Chunk(
                    id=obj_id,
                    doc_type="object",
                    content=synthesize_object_text(obj_for_text),
                    metadata={
                        "room_id": room_id,
                        "object_id": obj_id,
                        "name": obj_for_text["name"],
                        "has_vlm_description": bool(vlm),
                    },
                )
                if obj_id in seen_obj_ids:
                    # Replace existing room-VLM chunk with richer node-pickle version
                    chunks = [
                        c for c in chunks if c.metadata.get("object_id") != obj_id
                    ]
                chunks.append(new_chunk)
                seen_obj_ids.add(obj_id)

    return chunks


def normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return arr
    arr = arr.astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def prepare_text_query(gpt_interface, query: str, model_name: str) -> np.ndarray:
    vec = gpt_interface.embed_text([query], model=model_name)
    q = np.asarray(vec, dtype="float32")
    q = normalize_embeddings(q)
    return q


def combine_search_results(
    *,
    chunks: Sequence[Chunk],
    text_results: Optional[Tuple[np.ndarray, np.ndarray]],
    visual_results: Optional[Tuple[np.ndarray, np.ndarray]],
    frame_visual_chunk_indices: Sequence[int],
    object_visual_results: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    object_visual_chunk_indices: Optional[Sequence[int]] = None,
    doc_type_set: Optional[Set[str]],
    top_k: int,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return top_k results per modality separately.

    Returns:
        Dict with keys 'text', 'frame_visual', 'object_visual' containing
        lists of (chunk_index, score) tuples for each modality.
    """
    results = {}

    # Text results
    if text_results is not None:
        text_scores = []
        dists, inds = text_results
        for sim, idx in zip(dists[0], inds[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            if doc_type_set and chunks[idx].doc_type.lower() not in doc_type_set:
                continue
            text_scores.append((idx, float(sim)))
        results["text"] = sorted(text_scores, key=lambda x: x[1], reverse=True)[:top_k]

    # Frame visual results
    if visual_results is not None:
        frame_scores = []
        vdists, vinds = visual_results
        for vs, vi in zip(vdists[0], vinds[0]):
            if vi < 0 or vi >= len(frame_visual_chunk_indices):
                continue
            cidx = frame_visual_chunk_indices[vi]
            if cidx < 0 or cidx >= len(chunks):
                continue
            if doc_type_set and chunks[cidx].doc_type.lower() not in doc_type_set:
                continue
            frame_scores.append((cidx, float(vs)))
        results["frame_visual"] = sorted(
            frame_scores, key=lambda x: x[1], reverse=True
        )[:top_k]

    # Object visual results
    if object_visual_results is not None and object_visual_chunk_indices is not None:
        object_scores = []
        odists, oinds = object_visual_results
        for oscore, oi in zip(odists[0], oinds[0]):
            if oi < 0 or oi >= len(object_visual_chunk_indices):
                continue
            cidx = object_visual_chunk_indices[oi]
            if cidx < 0 or cidx >= len(chunks):
                continue
            if doc_type_set and chunks[cidx].doc_type.lower() not in doc_type_set:
                continue
            object_scores.append((cidx, float(oscore)))
        results["object_visual"] = sorted(
            object_scores, key=lambda x: x[1], reverse=True
        )[:top_k]

    return results
