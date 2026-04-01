"""Utility functions for scene description."""

from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import open3d as o3d


def normalize_tags(tags: Iterable[str], max_items: int = 80) -> List[str]:
    """Normalize and deduplicate tags."""
    seen = set()
    result = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
        if len(result) >= max_items:
            break
    return result


def coerce_valid_json(obj: Any) -> Dict[str, Any]:
    """Ensure a dictionary with expected fields."""
    base = {
        "caption": None,
        "room_type_guess": None,
        "description": None,
        "scene_layout": None,
        "objects": [],
    }
    if isinstance(obj, dict):
        merged = {**base, **obj}
        if not isinstance(merged.get("objects"), list):
            merged["objects"] = []
        return merged
    if isinstance(obj, str):
        base["description"] = obj
        return base
    return base


def parse_json_best_effort(text: str) -> Optional[Any]:
    """Parse JSON if possible, extracting from text if needed."""
    if not isinstance(text, str):
        return None

    s = text.strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract JSON object/array
    for start_char in ("{", "["):
        idx = s.find(start_char)
        if idx != -1:
            try:
                return json.loads(s[idx:])
            except Exception:
                continue
    return None


def sanitize_for_json(obj: Any) -> Any:
    """Sanitize object for JSON serialization."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return str(obj)


def is_pcd_visible_in_frame(
    pcd: o3d.geometry.PointCloud,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1000.0,
    depth_tolerance: float = 0.05,
    min_visible_ratio: float = 0.5,
) -> bool:
    """
    Check if a point cloud is visible in a camera frame.

    Args:
        pcd: Point cloud to check
        depth: Depth image
        pose: Camera pose (4x4 transform)
        intrinsics: Camera intrinsic matrix (3x3)
        depth_scale: Scale factor for depth values
        depth_tolerance: Tolerance for depth comparison
        min_visible_ratio: Minimum ratio of visible points

    Returns:
        True if the point cloud is sufficiently visible
    """
    if len(pcd.points) == 0:
        return False

    height, width = depth.shape
    depth_meters = depth.astype(np.float32) / depth_scale

    # Transform points to camera frame
    world_to_cam = np.linalg.inv(pose)
    points = np.asarray(pcd.points)
    points_h = np.hstack((points, np.ones((len(points), 1))))
    points_cam = (world_to_cam @ points_h.T).T[:, :3]

    # Filter points behind camera
    front_mask = points_cam[:, 2] > 0.1
    if not np.any(front_mask):
        return False

    points_cam = points_cam[front_mask]

    # Project to image
    projected = (intrinsics @ points_cam.T).T
    px = (projected[:, 0] / projected[:, 2]).astype(int)
    py = (projected[:, 1] / projected[:, 2]).astype(int)

    # Filter points in image bounds
    bounds_mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(bounds_mask):
        return False

    px_in, py_in = px[bounds_mask], py[bounds_mask]
    z_in = points_cam[bounds_mask, 2]

    # Check depth visibility
    depth_values = depth_meters[py_in, px_in]
    valid_depth = depth_values > 0
    if not np.any(valid_depth):
        return False

    depth_diff = np.abs(z_in[valid_depth] - depth_values[valid_depth])
    visible_count = np.sum(depth_diff < depth_tolerance)

    return (visible_count / len(pcd.points)) >= min_visible_ratio
