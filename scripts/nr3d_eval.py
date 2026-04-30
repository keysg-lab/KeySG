#!/usr/bin/env python3
"""
Nr3D evaluation script for KeySG (with RAG).

This script mirrors the logic in the nr3d_eval notebook:
- Loads scene annotations and GT scene description JSON
- Runs queries against KeySG with RAG (graph.query)
- Computes IoU@0.1 / IoU@0.25 metrics and prints per-split stats

Usage:
  python scripts/nr3d_eval.py \
      --scene_dir output/pipeline_scannet/ScanNet/scene0222_00 \

  # Only KeySG with RAG
  python scripts/nr3d_eval.py --scene_dir 
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from loguru import logger

# Local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from keysg.rag.graph_context_retriever import GraphContextRetriever, SearchResult
from keysg.rag.query_analysis import (
    analyze_and_expand_query,
    QueryAnalysisResult,
    _QuerySchema,
    SYSTEM_INSTRUCTIONS as _QUERY_ANALYSIS_INSTRUCTIONS,
)
from keysg.utils.load_utils import load_scene_nodes, get_objects

from eval_helpers import (
    # Geometry / BBox
    construct_bbox_corners,
    box3d_iou,
    _safe_bbox_from_center_extent,
    _bbox_to_np,
    _extract_bbox_corners,
    # Scene / Data loading
    _scene_base,
    _load_json,
    _load_scene_annotations,
    _load_gt_scene_objects,
    # RAG helpers
    _z_score_normalize,
    _get_obj_center,
    _compute_scene_center,
    _rank_frame_ids,
    _build_spatial_relations,
    _load_frame_images,
    # LLM batch
    _run_structured_batch as _run_structured_batch_rag,
    # Evaluation / Metrics
    _EVAL_KEYS,
    _evaluate_results,
    _format_metrics,
    # Debug logging
    _write_debug_entry,
    # Experiment tracking
    _collect_failed_queries,
    _save_experiment_artifacts,
    _write_outputs,
)

try:
    from models.llm.openai_api import GPTInterface
except ImportError:
    GPTInterface = None

try:
    from pydantic import BaseModel, Field as PydanticField
except ImportError:
    BaseModel = None
    PydanticField = None


# -- LLM-based object selection (mirrors notebook llm_query) --
if BaseModel is not None and PydanticField is not None:

    class ObjectSelection(BaseModel):
        object_id: Optional[str] = PydanticField(
            description="Chosen object ID from candidates; null if none match"
        )
        reason: str = PydanticField(description="Concise rationale for selection")
        confidence: float = PydanticField(
            ge=0, le=1, description="Calibrated confidence 0-1"
        )
        rejected_ids: List[str] = PydanticField(
            default_factory=list, description="IDs considered but rejected"
        )
        guess_id: Optional[str] = PydanticField(
            default=None, description="Closest guess if no confident selection"
        )

else:
    ObjectSelection = None  # type: ignore[misc,assignment]

_OBJECT_SELECTION_SYSTEM_PROMPT = (
    "You are a spatial reasoning expert tasked with selecting the single best `object_id` from a list of candidates "
    "that satisfies the USER QUERY. Use the provided frame descriptions and images (if available) to visually ground your decision.\n\n"
    "Decision Logic:\n"
    "1. **Semantic Match:** Identify candidates matching the target object's category and visual attributes.\n"
    "2. **Spatial Verification:** Filter by spatial constraints (e.g., 'left of', 'on top of') and proximity (use L2 distance/shortest path if provided).\n"
    "3. **Visual Confirmation:** Use frame context to resolve ambiguities (e.g., occlusion, specific relative position).\n"
    "4. **Selection:** Pick the ID with the highest cumulative evidence. If candidates are ambiguous, select the closest guess but assign low confidence.\n\n"
    "Output Requirements:\n"
    "- **Format:** Respond ONLY in the enforced JSON schema.\n"
    "- **ID Validity:** Never hallucinate IDs. Only use IDs from the Candidate List or Frame Description or Image.\n"
    "- **Confidence:**\n"
    "  - High (~0.9): Unique, strong alignment.\n"
    "  - Medium (0.6-0.75): Good match but potential uncertainty.\n"
    "  - Low (≤0.35): Ambiguous guess or no plausible candidate (return null if completely irrelevant).\n"
    "- **Justification:** Briefly cite specific candidate attributes, spatial relations, and frame context that drove the decision."
)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline: KeySG RAG
# ──────────────────────────────────────────────────────────────────────────────


def _run_keysg_rag(
    scene_dir: str,
    annotations: List[Dict[str, Any]],
    *,
    limit: Optional[int],
    include_frame_images: bool = True,
    include_frame_text: bool = True,
    max_frame_images: int = 2,
    rag_model: str = "gpt-5-mini",
    top_k_objects: int = 10,
    top_k_frames: int = 1,
    gt_corners_map: Optional[Dict[str, np.ndarray]] = None,
    gt_label_map: Optional[Dict[str, str]] = None,
    debug_log_path: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Run KeySG RAG pipeline with batched LLM calls.

    Pipeline:
      Phase 1 — For each query: analyze, retrieve candidates/frames, build context
      Phase 2 — Batch LLM structured prompts (batch_size at a time)
      Phase 3 — Process results: bbox lookup, debug log, output dicts
    """
    if GPTInterface is None:
        raise ImportError("GPTInterface required. Install openai package.")
    if ObjectSelection is None:
        raise ImportError("pydantic required. Install pydantic package.")

    # -- Setup --
    logger.info("Building RAG retriever from {}", scene_dir)
    retriever = GraphContextRetriever(scene_dir)
    retriever.build_chunks()
    retriever.compute_embeddings(
        compute_frame_visual=True,
        compute_object_visual=True,
    )
    retriever.build_faiss_index()

    objects = getattr(retriever, "objects", None) or []
    if not objects:
        try:
            nodes = load_scene_nodes(scene_dir)
            objects = get_objects(nodes)
        except Exception as e:
            logger.warning("Failed to load scene objects: {}", e)
            objects = []

    scene_center = _compute_scene_center(objects)
    obj_by_id = {str(o.id): o for o in objects}
    gpt = GPTInterface()
    chunk_map = {c.id: c for c in retriever.chunks}

    total = len(annotations) if limit is None else min(limit, len(annotations))
    ann_slice = annotations[:total]

    # ── Phase 1a: Batch query analysis ──
    valid_anns: List[Dict[str, Any]] = []
    utterances: List[str] = []
    for ann in ann_slice:
        utterance = ann.get("utterance")
        if utterance:
            valid_anns.append(ann)
            utterances.append(utterance)

    logger.info("Phase 1a: Batch query analysis for {} queries", len(utterances))
    analysis_prompts = [f"User query: {u}" for u in utterances]
    analysis_results = _run_structured_batch_rag(
        gpt,
        analysis_prompts,
        images_list=None,
        response_model=_QuerySchema,
        model="gpt-5-nano",
        instructions=_QUERY_ANALYSIS_INSTRUCTIONS,
        reasoning_effort="low",
    )

    # Parse analysis results
    analyses: List[Dict[str, Any]] = []
    for i, result in enumerate(analysis_results):
        if isinstance(result, Exception):
            logger.warning(
                "Query analysis failed for '{}': {}", utterances[i][:50], result
            )
            analyses.append(
                {
                    "target_object": utterances[i],
                    "anchor_objects": [],
                    "relation_polarity": None,
                }
            )
        else:
            parsed = result.model_dump()
            analyses.append(
                {
                    "target_object": parsed.get("target_object"),
                    "anchor_objects": parsed.get("anchor_objects") or [],
                    "relation_polarity": parsed.get("relation_polarity"),
                }
            )

    # ── Phase 1b: RAG retrieval + context building ──
    logger.info("Phase 1b: Building context for {} queries", len(valid_anns))
    prepared: List[Dict[str, Any]] = []
    iterable = tqdm(
        zip(valid_anns, utterances, analyses),
        desc="RAG retrieval",
        unit="q",
        total=len(valid_anns),
    )
    for ann, utterance, analysis in iterable:
        target_object = analysis["target_object"]
        anchor_objects = analysis["anchor_objects"]
        relation_polarity = analysis["relation_polarity"]

        # 1. Retrieve candidates
        search_query = target_object or utterance
        target_results = retriever.search(
            search_query,
            top_k=top_k_objects,
            doc_types=["object"],
            object_modality="both",
        )

        anchor_query = " ".join(anchor_objects) if anchor_objects else ""
        if anchor_query:
            anchor_results = retriever.search(
                anchor_query,
                top_k=top_k_objects,
                doc_types=["object"],
                object_modality="both",
            )
        else:
            anchor_results = {"object_visual": [], "text": []}

        # 2. Retrieve and rank frames
        frame_results = retriever.search(
            utterance,
            top_k=top_k_frames,
            doc_types=["frame"],
            object_modality="both",
            frame_modality="both",
        )
        top_frame_ids = _rank_frame_ids(
            frame_results, top_k_frames, include_text=False, include_visual=True
        )
        top_frame_chunks = [chunk_map[fid] for fid in top_frame_ids if fid in chunk_map]

        # 3. Build context text (fuse text + visual scores per object)
        # target_vis = _fuse_obj_results(target_results, top_k_objects)
        # anchor_vis = _fuse_obj_results(anchor_results, top_k_objects)
        # target_vis = target_results.get("text", [])
        # anchor_vis = anchor_results.get("text", [])
        target_vis = target_results.get("object_visual", [])
        anchor_vis = anchor_results.get("object_visual", [])

        sections = [
            f"USER QUERY: {utterance}",
            f"PARSED TARGET: {target_object}",
            f"PARSED ANCHORS: {anchor_objects}",
        ]
        if target_vis:
            lines = ["Target Object Candidates:"]
            for i, r in enumerate(target_vis):
                lines.append(f"{i+1}. ID={r.chunk.id}). Desc={r.chunk.content}")
            sections.append("\n".join(lines))
        if anchor_vis:
            lines = ["Anchor Object Candidates:"]
            for i, r in enumerate(anchor_vis):
                lines.append(f"{i+1}. ID={r.chunk.id}. Desc={r.chunk.content}")
            sections.append("\n".join(lines))

        if include_frame_text and top_frame_chunks:
            lines = ["Relevant Frames:"]
            for i, c in enumerate(top_frame_chunks):
                lines.append(f"{i+1}. FRAME_ID={c.id}. {c.content}")
            sections.append("\n".join(lines))

        # Only add spatial relations when the query has distance polarity
        spatial_rel_lines: List[str] = []
        if target_vis and anchor_vis and relation_polarity:
            spatial_rel_lines = _build_spatial_relations(
                target_vis, anchor_vis, obj_by_id, scene_center
            )
            if spatial_rel_lines:
                sections.append(
                    "Spatial Relations (target <-> anchor):\n"
                    + "\n".join(spatial_rel_lines)
                )

        context_text = "\n\n".join(sections)

        # 4. Load frame images (optional)
        images = None
        if include_frame_images:
            images = _load_frame_images(top_frame_chunks, max_frame_images)

        prepared.append(
            {
                "ann": ann,
                "utterance": utterance,
                "target_object": target_object,
                "anchor_objects": anchor_objects,
                "target_vis": target_vis,
                "anchor_vis": anchor_vis,
                "spatial_rel_lines": spatial_rel_lines,
                "frame_results": frame_results,
                "context_text": context_text,
                "images": images,
            }
        )

    # ── Phase 2: Batched LLM calls ──
    logger.info("Phase 2: Running LLM selection in batches of {}", batch_size)
    all_selections: List[Any] = [None] * len(prepared)

    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        prompts = [p["context_text"] for p in batch]
        images_list = [p["images"] for p in batch]

        batch_selections = _run_structured_batch_rag(
            gpt,
            prompts,
            images_list,
            response_model=ObjectSelection,
            model=rag_model,
            instructions=_OBJECT_SELECTION_SYSTEM_PROMPT,
            reasoning_effort="medium",
            detail="high",
        )

        for i, sel in enumerate(batch_selections):
            all_selections[batch_start + i] = sel

        logger.info(
            "  Batch {}-{}/{} done",
            batch_start + 1,
            min(batch_start + batch_size, len(prepared)),
            len(prepared),
        )

    # ── Phase 3: Process results ──
    logger.info("Phase 3: Processing results")
    debug_file = None
    if debug_log_path:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        debug_file = open(debug_log_path, "w")
        debug_file.write(f"# Debug log — {datetime.utcnow().isoformat()}Z\n")
        debug_file.write(f"# Scene: {scene_dir}\n")
        debug_file.write(f"# Model: {rag_model}\n\n")

    results: List[Dict[str, Any]] = []
    for query_idx, (p, selection) in enumerate(zip(prepared, all_selections), start=1):
        ann = p["ann"]
        utterance = p["utterance"]

        # Handle LLM errors
        if isinstance(selection, Exception):
            logger.warning("LLM error for ann_id={}: {}", ann.get("ann_id"), selection)
            pred_id = None
        else:
            pred_id = selection.object_id
            if pred_id is None:
                pred_id = getattr(selection, "guess_id", None)

        # Get bbox from scene objects
        bbox = None
        if pred_id is not None:
            obj = next((o for o in objects if str(o.id) == str(pred_id)), None)
            if obj is not None:
                bbox = _extract_bbox_corners(obj)

        # Debug log
        if debug_file is not None and not isinstance(selection, Exception):
            _write_debug_entry(
                debug_file,
                query_idx,
                ann,
                utterance,
                p["context_text"],
                selection,
                pred_id,
                bbox,
                gt_corners_map,
                gt_label_map,
                frame_results=p["frame_results"],
                images=p["images"],
            )

        results.append(
            {
                "ann_id": ann.get("ann_id"),
                "utterance": utterance,
                "ground_truth_target_id": ann.get("target_id"),
                "predicted_object_id": pred_id,
                "bbox_3d": bbox.tolist() if bbox is not None else None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    if debug_file is not None:
        debug_file.close()
        logger.info("Debug log written to {}", debug_log_path)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Nr3D evaluation for KeySG with RAG")
    parser.add_argument(
        "--scene_dir", type=str, required=True, help="KeySG pipeline output directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/experiments/nr3d_eval",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--include_frame_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include frame images for visual grounding in LLM queries",
    )
    parser.add_argument(
        "--include_frame_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include top-k frame text descriptions in RAG context",
    )
    parser.add_argument(
        "--rag_model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model for RAG LLM object selection",
    )
    parser.add_argument(
        "--max_frame_images",
        type=int,
        default=4,
        help="Max frame images to include in LLM prompt",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of queries"
    )
    parser.add_argument(
        "--iou_thresholds",
        type=str,
        default="0.001,0.1,0.25",
        help="Comma-separated IoU thresholds",
    )
    parser.add_argument(
        "--top_k_objects",
        type=int,
        default=10,
        help="Top-K object candidates to retrieve per query",
    )
    parser.add_argument(
        "--top_k_frames",
        type=int,
        default=10,
        help="Top-K frame candidates to retrieve per query",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for LLM queries (concurrent async calls)",
    )
    parser.add_argument(
        "--nr3d_root",
        type=str,
        default="/mnt/ssd2/datasets/ScanNetv2/NR3D",
        help="Root directory containing Nr3D annotations and GT",
    )
    parser.add_argument(
        "--debug_log",
        type=str,
        default=None,
        help="Path to write per-query debug log (default: <output_dir>/<scene>_debug.log)",
    )
    args = parser.parse_args()

    iou_thresholds = tuple(
        float(x) for x in args.iou_thresholds.split(",") if x.strip()
    )

    annotations = _load_scene_annotations(args.scene_dir, nr3d_root=args.nr3d_root)
    gt_objects = _load_gt_scene_objects(args.scene_dir, nr3d_root=args.nr3d_root)

    gt_corners_map = {}
    for ob in gt_objects:
        try:
            gt_corners_map[str(ob["id"])] = construct_bbox_corners(
                ob["bbox_center"], ob["bbox_extent"]
            )
        except Exception:
            continue

    gt_label_map = {
        str(ob.get("id", "")): ob.get("label", ob.get("category", "?"))
        for ob in gt_objects
        if ob.get("id")
    }

    base = _scene_base(args.scene_dir)
    debug_log_path = args.debug_log or os.path.join(
        args.output_dir, f"{base}_debug.log"
    )

    _save_experiment_artifacts(args.output_dir, args, __file__)

    rag_results = _run_keysg_rag(
        args.scene_dir,
        annotations,
        limit=args.limit,
        include_frame_images=args.include_frame_images,
        include_frame_text=args.include_frame_text,
        max_frame_images=args.max_frame_images,
        rag_model=args.rag_model,
        top_k_frames=args.top_k_frames,
        top_k_objects=args.top_k_objects,
        gt_corners_map=gt_corners_map,
        gt_label_map=gt_label_map,
        debug_log_path=debug_log_path,
        batch_size=args.batch_size,
    )
    rag_metrics = _evaluate_results(
        rag_results, annotations, gt_corners_map, iou_thresholds
    )
    _write_outputs(
        args.output_dir,
        args.scene_dir,
        "keysg_rag",
        rag_results,
        rag_metrics,
        annotations,
        gt_corners_map,
    )


if __name__ == "__main__":
    main()
