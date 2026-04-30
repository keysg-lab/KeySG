from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger


_EVAL_KEYS = (
    "mentions_target_class",
    "uses_object_lang",
    "uses_spatial_lang",
    "uses_color_lang",
    "uses_shape_lang",
    "is_easy",
    "is_view_dep",
)


def construct_bbox_corners(center: Iterable[float], box_size: Iterable[float]) -> np.ndarray:
    cx, cy, cz = [float(x) for x in center]
    sx, sy, sz = [float(x) for x in box_size]
    corners = np.array(
        [
            [sx / 2, sy / 2, sz / 2],
            [sx / 2, -sy / 2, sz / 2],
            [-sx / 2, -sy / 2, sz / 2],
            [-sx / 2, sy / 2, sz / 2],
            [sx / 2, sy / 2, -sz / 2],
            [sx / 2, -sy / 2, -sz / 2],
            [-sx / 2, -sy / 2, -sz / 2],
            [-sx / 2, sy / 2, -sz / 2],
        ],
        dtype=np.float32,
    )
    corners[:, 0] += cx
    corners[:, 1] += cy
    corners[:, 2] += cz
    return corners


def get_box3d_min_max(corners: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    arr = _bbox_to_np(corners)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].min()),
        float(arr[:, 1].max()),
        float(arr[:, 2].min()),
        float(arr[:, 2].max()),
    )


def box3d_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)

    x_a = max(x_min_1, x_min_2)
    y_a = max(y_min_1, y_min_2)
    z_a = max(z_min_1, z_min_2)
    x_b = min(x_max_1, x_max_2)
    y_b = min(y_max_1, y_max_2)
    z_b = min(z_max_1, z_max_2)

    inter_vol = max(x_b - x_a, 0.0) * max(y_b - y_a, 0.0) * max(z_b - z_a, 0.0)
    vol_1 = max(x_max_1 - x_min_1, 0.0) * max(y_max_1 - y_min_1, 0.0) * max(z_max_1 - z_min_1, 0.0)
    vol_2 = max(x_max_2 - x_min_2, 0.0) * max(y_max_2 - y_min_2, 0.0) * max(z_max_2 - z_min_2, 0.0)
    return float(inter_vol / (vol_1 + vol_2 - inter_vol + 1e-8))


def _bbox_to_np(bbox: Any) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"Expected bbox corners with shape (8, 3), got {arr.shape}")
    return arr


def _safe_bbox_from_center_extent(center: Any, extent: Any) -> Optional[np.ndarray]:
    try:
        return construct_bbox_corners(center, extent)
    except Exception:
        return None


def _extract_bbox_corners(obj: Any) -> Optional[np.ndarray]:
    bbox = getattr(obj, "bbox_3d", None)
    if bbox is not None:
        if hasattr(bbox, "min_bound") and hasattr(bbox, "max_bound"):
            mn = np.asarray(bbox.min_bound, dtype=np.float32)
            mx = np.asarray(bbox.max_bound, dtype=np.float32)
            return construct_bbox_corners((mn + mx) / 2.0, mx - mn)
        try:
            arr = np.asarray(bbox, dtype=np.float32)
            if arr.shape == (8, 3):
                return arr
        except Exception:
            pass

    center = getattr(obj, "bbox_center", None)
    extent = getattr(obj, "bbox_extent", None)
    if center is not None and extent is not None:
        return _safe_bbox_from_center_extent(center, extent)

    pcd = getattr(obj, "pcd", None)
    if pcd is not None:
        try:
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.size:
                mn = pts.min(axis=0)
                mx = pts.max(axis=0)
                return construct_bbox_corners((mn + mx) / 2.0, mx - mn)
        except Exception:
            pass
    return None


def _scene_base(scene_dir: str) -> str:
    return os.path.basename(scene_dir.rstrip("/"))


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _find_nr3d_file(nr3d_root: str, filename: str) -> Optional[str]:
    direct = os.path.join(nr3d_root, filename)
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(nr3d_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _load_scene_annotations(scene_dir: str, nr3d_root: Optional[str] = None) -> List[Dict[str, Any]]:
    base = _scene_base(scene_dir)
    ann_path = os.path.join(scene_dir, f"{base}_annotation.json")
    if not os.path.isfile(ann_path) and nr3d_root:
        alt = _find_nr3d_file(nr3d_root, f"{base}_annotation.json")
        if alt:
            ann_path = alt
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    return _load_json(ann_path)


def _load_gt_scene_objects(scene_dir: str, nr3d_root: Optional[str] = None) -> List[Dict[str, Any]]:
    base = _scene_base(scene_dir)
    gt_path = os.path.join(scene_dir, f"{base}_scene_description_unaligned.json")
    if not os.path.isfile(gt_path) and nr3d_root:
        alt = _find_nr3d_file(nr3d_root, f"{base}_scene_description_unaligned.json")
        if alt:
            gt_path = alt
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT scene description file not found: {gt_path}")
    return _load_json(gt_path)


def _z_score_normalize(values: Iterable[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return []
    std = float(arr.std())
    if std < 1e-8:
        return [0.0] * int(arr.size)
    mean = float(arr.mean())
    return [float((x - mean) / std) for x in arr]


def _get_obj_center(obj: Any) -> Optional[np.ndarray]:
    bbox = _extract_bbox_corners(obj)
    if bbox is not None:
        return bbox.mean(axis=0)
    return None


def _compute_scene_center(objects: List[Any]) -> Optional[np.ndarray]:
    centers = [c for c in (_get_obj_center(o) for o in objects) if c is not None]
    if not centers:
        return None
    return np.mean(np.stack(centers, axis=0), axis=0)


def _rank_frame_ids(
    frame_results: Dict[str, List[Any]],
    top_k: int,
    *,
    include_text: bool = False,
    include_visual: bool = True,
) -> List[str]:
    scores: Dict[str, float] = {}
    modalities: List[str] = []
    if include_text:
        modalities.append("text")
    if include_visual:
        modalities.append("frame_visual")

    for modality in modalities:
        results = frame_results.get(modality, []) or []
        norm_scores = _z_score_normalize([r.score for r in results])
        for r, score in zip(results, norm_scores):
            scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:top_k]]


def _build_spatial_relations(
    target_results: List[Any],
    anchor_results: List[Any],
    obj_by_id: Dict[str, Any],
    scene_center: Optional[np.ndarray],
) -> List[str]:
    lines: List[str] = []
    for t_res in target_results[:5]:
        t_obj = obj_by_id.get(str(t_res.chunk.id))
        t_center = _get_obj_center(t_obj) if t_obj is not None else None
        if t_center is None:
            continue
        for a_res in anchor_results[:5]:
            a_obj = obj_by_id.get(str(a_res.chunk.id))
            a_center = _get_obj_center(a_obj) if a_obj is not None else None
            if a_center is None:
                continue
            delta = t_center - a_center
            dist = float(np.linalg.norm(delta))
            parts = [
                f"target={t_res.chunk.id}",
                f"anchor={a_res.chunk.id}",
                f"distance={dist:.3f}",
                f"dx={delta[0]:.3f}",
                f"dy={delta[1]:.3f}",
                f"dz={delta[2]:.3f}",
            ]
            if scene_center is not None:
                parts.append(
                    f"target_scene_dist={float(np.linalg.norm(t_center - scene_center)):.3f}"
                )
                parts.append(
                    f"anchor_scene_dist={float(np.linalg.norm(a_center - scene_center)):.3f}"
                )
            lines.append(", ".join(parts))
    return lines


def _load_frame_images(top_frame_chunks: List[Any], max_frame_images: int) -> Optional[List[Image.Image]]:
    images: List[Image.Image] = []
    for chunk in top_frame_chunks:
        md = getattr(chunk, "metadata", {}) or {}
        path = md.get("labeled_image_path") or md.get("image_path")
        if not path or not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        except Exception as exc:
            logger.warning("Failed to load frame image {}: {}", path, exc)
        if len(images) >= max_frame_images:
            break
    return images or None


def _run_structured_batch(
    gpt: Any,
    prompts: List[str],
    images_list: Optional[List[Optional[Any]]],
    *,
    response_model: Any,
    model: str,
    instructions: str,
    reasoning_effort: str = "medium",
    detail: str = "auto",
) -> List[Any]:
    images = images_list if images_list is not None else None
    return asyncio.run(
        gpt.structured_prompt_batch(
            prompts=prompts,
            response_model=response_model,
            model=model,
            images=images,
            instructions=instructions,
            reasoning_effort=reasoning_effort,
            detail=detail,
        )
    )


def _evaluate_results(
    results: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
    iou_thresholds: Tuple[float, ...],
    *,
    legacy_group_metrics_bug: bool = False,
) -> Dict[str, Any]:
    ann_by_id = {ann["ann_id"]: ann for ann in annotations if "ann_id" in ann}
    overall_hits = {thr: 0 for thr in iou_thresholds}
    overall_total = 0
    by_key: Dict[str, Dict[bool, Dict[str, Any]]] = {}

    for key in _EVAL_KEYS:
        by_key[key] = {
            False: {"total": 0, "hits": {thr: 0 for thr in iou_thresholds}},
            True: {"total": 0, "hits": {thr: 0 for thr in iou_thresholds}},
        }

    for r in results:
        ann = ann_by_id.get(r.get("ann_id"))
        if not ann:
            continue
        gt_id = str(ann.get("target_id"))
        gt_bbox = gt_corners_map.get(gt_id)
        pred_bbox = _bbox_to_np(r["bbox_3d"]) if r.get("bbox_3d") is not None else None
        iou = float(box3d_iou(pred_bbox, gt_bbox)) if pred_bbox is not None and gt_bbox is not None else 0.0
        r["iou"] = iou

        overall_total += 1
        for thr in iou_thresholds:
            if iou >= thr:
                overall_hits[thr] += 1

        if not (legacy_group_metrics_bug and not ann.get("ann_id")):
            for key in _EVAL_KEYS:
                value = bool(ann.get(key, False))
                bucket = by_key[key][value]
                bucket["total"] += 1
                for thr in iou_thresholds:
                    if iou >= thr:
                        bucket["hits"][thr] += 1

    return {
        "iou_thresholds": iou_thresholds,
        "overall": {"total": overall_total, "hits": overall_hits},
        "by_key": by_key,
    }


def _format_metrics(metrics: Dict[str, Any]) -> List[str]:
    thresholds = tuple(metrics["iou_thresholds"])
    lines = [f"Evaluation Results (IoU thresholds: {thresholds})"]
    for key in _EVAL_KEYS:
        for value in (False, True):
            bucket = metrics["by_key"][key][value]
            total = int(bucket["total"])
            if total == 0:
                score_bits = [f"IoU@{thr}:N/A" for thr in thresholds]
            else:
                score_bits = [
                    f"IoU@{thr}:{100.0 * bucket['hits'][thr] / total:.1f}%"
                    for thr in thresholds
                ]
            lines.append(
                f"{key:<24} {str(value):<5} -> {', '.join(score_bits)} (n={total})"
            )

    overall_total = int(metrics["overall"]["total"])
    for thr in thresholds:
        acc = 0.0 if overall_total == 0 else 100.0 * metrics["overall"]["hits"][thr] / overall_total
        lines.append(f"OVERALL IoU@{thr}: {acc:.1f}% (n={overall_total})")
    return lines


def _write_debug_entry(
    debug_file: Any,
    query_idx: int,
    ann: Dict[str, Any],
    utterance: str,
    context_text: str,
    selection: Any,
    pred_id: Optional[str],
    bbox: Optional[np.ndarray],
    gt_corners_map: Optional[Dict[str, np.ndarray]],
    gt_label_map: Optional[Dict[str, str]],
    *,
    frame_results: Optional[Dict[str, List[Any]]] = None,
    images: Optional[List[Any]] = None,
) -> None:
    gt_id = str(ann.get("target_id"))
    gt_bbox = gt_corners_map.get(gt_id) if gt_corners_map else None
    iou = box3d_iou(bbox, gt_bbox) if bbox is not None and gt_bbox is not None else None
    gt_label = gt_label_map.get(gt_id, "?") if gt_label_map else "?"

    debug_file.write(f"## Query {query_idx}\n")
    debug_file.write(f"ann_id: {ann.get('ann_id')}\n")
    debug_file.write(f"utterance: {utterance}\n")
    debug_file.write(f"ground_truth_target_id: {gt_id}\n")
    debug_file.write(f"ground_truth_label: {gt_label}\n")
    debug_file.write(f"predicted_object_id: {pred_id}\n")
    debug_file.write(f"selection_confidence: {getattr(selection, 'confidence', None)}\n")
    debug_file.write(f"selection_reason: {getattr(selection, 'reason', '')}\n")
    debug_file.write(f"guess_id: {getattr(selection, 'guess_id', None)}\n")
    debug_file.write(f"rejected_ids: {getattr(selection, 'rejected_ids', [])}\n")
    debug_file.write(f"iou: {iou if iou is not None else 'N/A'}\n")
    if frame_results:
        for modality, results in frame_results.items():
            top = [(r.chunk.id, float(r.score)) for r in (results or [])[:5]]
            debug_file.write(f"{modality}_top5: {top}\n")
    debug_file.write(f"image_count: {0 if images is None else len(images)}\n")
    debug_file.write("context:\n")
    debug_file.write(context_text)
    debug_file.write("\n\n")


def _collect_failed_queries(
    results: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    ann_by_id = {ann["ann_id"]: ann for ann in annotations if "ann_id" in ann}
    failed: List[Dict[str, Any]] = []
    for r in results:
        ann_id = r.get("ann_id")
        ann = ann_by_id.get(ann_id)
        if not ann:
            continue
        gt_id = str(ann.get("target_id"))
        pred_bbox = _bbox_to_np(r["bbox_3d"]) if r.get("bbox_3d") is not None else None
        iou = float(r.get("iou", 0.0))
        if iou == 0.0 and pred_bbox is not None and gt_id in gt_corners_map:
            iou = box3d_iou(pred_bbox, gt_corners_map[gt_id])
        if iou < iou_threshold:
            failed.append(
                {
                    "ann_id": ann_id,
                    "utterance": r.get("utterance", ann.get("utterance", "")),
                    "ground_truth_target_id": ann.get("target_id"),
                    "predicted_object_id": r.get("predicted_object_id"),
                    "iou": float(iou),
                    "mentions_target_class": ann.get("mentions_target_class"),
                    "uses_spatial_lang": ann.get("uses_spatial_lang"),
                    "uses_color_lang": ann.get("uses_color_lang"),
                    "uses_shape_lang": ann.get("uses_shape_lang"),
                }
            )
    return failed


def _save_experiment_artifacts(output_dir: str, args: argparse.Namespace, script_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    config = vars(args).copy()
    config["timestamp"] = datetime.utcnow().isoformat() + "Z"
    config["script"] = os.path.abspath(script_path)
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved experiment config to {}", config_path)

    script_name = os.path.splitext(os.path.basename(script_path))[0]
    script_copy_path = os.path.join(output_dir, f"{script_name}_snapshot.py")
    shutil.copy2(script_path, script_copy_path)
    logger.info("Saved script snapshot to {}", script_copy_path)


def _write_outputs(
    output_dir: str,
    scene_dir: str,
    method: str,
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    gt_corners_map: Dict[str, np.ndarray],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base = _scene_base(scene_dir)

    results_path = os.path.join(output_dir, f"{base}_{method}_results.json")
    metrics_path = os.path.join(output_dir, f"{base}_{method}_metrics.txt")

    with open(results_path, "w") as f:
        json.dump({"scene_dir": scene_dir, "method": method, "results": results}, f, indent=2)

    scene_id_match = re.search(r"\d+", base)
    scene_id = scene_id_match.group(0) if scene_id_match else "0"
    lines = [f"## {scene_id}:", f"SCENE: {base}"]
    lines.extend(_format_metrics(metrics))
    with open(metrics_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved results to {}", results_path)
    logger.info("Saved metrics to {}", metrics_path)

    failed = _collect_failed_queries(results, annotations, gt_corners_map, 0.1)
    if failed:
        failed_path = os.path.join(output_dir, f"{base}_{method}_failed.json")
        with open(failed_path, "w") as f:
            json.dump(
                {
                    "scene_dir": scene_dir,
                    "method": method,
                    "iou_threshold": 0.1,
                    "total_queries": len(results),
                    "failed_count": len(failed),
                    "failed_queries": failed,
                },
                f,
                indent=2,
            )
        logger.info("Saved {} failed queries (IoU<0.1) to {}", len(failed), failed_path)
