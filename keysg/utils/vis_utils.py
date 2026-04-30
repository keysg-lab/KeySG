"""
Visualization utilities for KeySG 3D scene graphs.

This module provides visualization functions for various types of point clouds
including nodes, rooms, floors, and functional elements.
"""

import os
import numpy as np
import open3d as o3d
import cv2
from typing import List, Optional, Union, Dict, Any, Tuple
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt

from keysg.scene_segmentor.obj_node import ObjNode


def generate_colors(num_colors: int) -> np.ndarray:
    """
    Generate distinct colors for visualization using the golden ratio.

    This method uses the golden angle to ensure that consecutive colors are
    maximally different from each other, which is better for visualization
    than a simple linear distribution of hues.

    Args:
        num_colors: Number of distinct colors needed

    Returns:
        Array of RGB colors, shape (num_colors, 3)
    """
    if num_colors == 0:
        return np.array([])

    # Use the golden ratio to generate well-spaced hues
    golden_ratio_conjugate = 0.618033988749895
    hues = np.arange(num_colors)
    hues = (hues * golden_ratio_conjugate) % 1.0

    colors = []
    for hue in hues:
        # Convert HSV to RGB (with S=1, V=1 for vibrant colors)
        rgb = plt.cm.hsv(hue)[:3]  # Take only RGB, ignore alpha
        colors.append(rgb)

    return np.array(colors)


def visualize_single_node(
    node: Any,  # ObjNode
    window_name: str = None,
    show_functional: bool = True,
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize a single object node's point cloud and masks.

    Args:
        node: ObjNode to visualize
        window_name: Optional window name for 3D visualization
        show_functional: Whether to show functional elements
        show_bbox: Whether to show bounding boxes
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if window_name is None:
        window_name = f"Object Node: {node.id}"

    geometries = []

    # Visualize main point cloud
    if node.pcd is not None and len(node.pcd.points) > 0:
        pcd_vis = o3d.geometry.PointCloud(node.pcd)
        pcd_vis.paint_uniform_color([0.0, 0.7, 0.0])  # Green for main object
        geometries.append(pcd_vis)

    # Visualize functional elements
    if (
        show_functional
        and hasattr(node, "functional_elements")
        and node.functional_elements
    ):
        for i, func_elem in enumerate(node.functional_elements):
            if (
                hasattr(func_elem, "pcd")
                and func_elem.pcd is not None
                and len(func_elem.pcd.points) > 0
            ):
                func_pcd = func_elem.pcd.__copy__()
                # Use red for functional elements
                func_pcd.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(func_pcd)

    # Save visualization if requested
    if save_path and geometries:
        save_visualization(geometries, save_path, save_format)

    # Show 3D visualization if we have point clouds
    if geometries:
        logger.info(f"Visualizing node {node.id} with {len(geometries)} geometries")
        o3d.visualization.draw_geometries(
            geometries, window_name=window_name, width=800, height=600
        )
    else:
        logger.warning(f"No point cloud data to visualize for node {node.id}")

    # Display 2D mask and RGB frame if available
    if (
        hasattr(node, "masks_2d")
        and hasattr(node, "rgb_frames")
        and node.masks_2d
        and node.rgb_frames
    ):
        visualize_2d_masks(node.masks_2d, node.rgb_frames, node.id, save_path)


def visualize_nodes_collection(
    nodes: List[Any],  # List[ObjNode]
    window_name: str = "Object Nodes Collection",
    show_functional: bool = True,
    show_bbox: bool = False,
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize a collection of object nodes' point clouds together.
    Each node gets a different color for distinction.

    Args:
        nodes: List of ObjNode objects to visualize
        window_name: Window name for 3D visualization
        show_functional: Whether to include functional element point clouds
        show_bbox: Whether to show bounding boxes
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if not nodes:
        logger.warning("No nodes to visualize")
        return

    logger.info(f"Visualizing {len(nodes)} object nodes...")

    geometries = []
    colors = generate_colors(len(nodes))

    nodes_with_pcd = 0
    total_points = 0

    for i, node in enumerate(nodes):
        node_color = colors[i] if i < len(colors) else [0.5, 0.5, 0.5]

        # Main point cloud
        if node.pcd is not None and len(node.pcd.points) > 0:
            pcd_vis = node.pcd.__copy__()
            pcd_vis.paint_uniform_color(node_color)
            geometries.append(pcd_vis)
            nodes_with_pcd += 1
            total_points += len(node.pcd.points)

            # Add bounding box if requested
            if show_bbox and hasattr(node, "bbox_3d") and node.bbox_3d is not None:
                bbox_vis = node.bbox_3d
                geometries.append(bbox_vis)

        # Functional elements with brighter versions of the same color
        if (
            show_functional
            and hasattr(node, "functional_elements")
            and node.functional_elements
        ):
            bright_color = np.minimum(node_color * 1.5, 1.0)  # Brighten the color
            for func_elem in node.functional_elements:
                if (
                    hasattr(func_elem, "pcd")
                    and func_elem.pcd is not None
                    and len(func_elem.pcd.points) > 0
                ):
                    func_pcd = func_elem.pcd.__copy__()
                    func_pcd.paint_uniform_color(bright_color)
                    geometries.append(func_pcd)

    # Save visualization if requested
    if save_path and geometries:
        save_visualization(geometries, save_path, save_format)

    if not geometries:
        logger.warning("No point clouds to visualize")
        return

    logger.info(f"Total visualization: {nodes_with_pcd} nodes, {total_points} points")
    logger.info("Opening 3D visualization...")
    logger.info(
        "Color legend: Each object has a unique color, functional elements are brighter versions"
    )

    o3d.visualization.draw_geometries(
        geometries, window_name=window_name, width=1200, height=800
    )


def visualize_functional_elements(
    nodes: List[Any],  # List[ObjNode]
    window_name: str = "Functional Elements",
    scene_pcd: o3d.geometry.PointCloud | None = None,
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize functional elements of the object nodes.

    Args:
        nodes: List of ObjNode objects
        window_name: Window name for 3D visualization
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if not nodes:
        logger.warning("No nodes provided for functional element visualization")
        return

    logger.info(f"Visualizing functional elements for {len(nodes)} object nodes...")

    geometries = []
    if scene_pcd:
        geometries.append(scene_pcd)

    for i, node in enumerate(nodes):
        if hasattr(node, "functional_elements") and node.functional_elements:
            for j, func_elem in enumerate(node.functional_elements):
                if (
                    hasattr(func_elem, "pcd")
                    and func_elem.pcd is not None
                    and len(func_elem.pcd.points) > 0
                ):
                    func_pcd = func_elem.pcd.__copy__()
                    func_pcd.paint_uniform_color(
                        [1.0, 0.0, 0.0]
                    )  # Red for functional elements
                    geometries.append(func_pcd)
        if "fun" in node.id.lower():
            if node.pcd is not None and len(node.pcd.points) > 0:
                func_pcd = node.pcd.__copy__()
                func_pcd.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(func_pcd)

    # Save visualization if requested
    if save_path and geometries:
        save_visualization(geometries, save_path, save_format)

    if not geometries:
        logger.warning("No functional elements to visualize")
        return

    logger.info(f"Total functional elements visualized: {len(geometries)}")
    logger.info("Opening 3D visualization for functional elements...")
    o3d.visualization.draw_geometries(
        geometries, window_name=window_name, width=1200, height=800
    )


def visualize_room_pcd(
    room_pcd: o3d.geometry.PointCloud,
    room_id: str,
    window_name: Optional[str] = None,
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize a room's point cloud.

    Args:
        room_pcd: Point cloud of the room
        room_id: Room identifier
        window_name: Optional window name
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if window_name is None:
        window_name = f"Room {room_id}"

    if room_pcd is None or len(room_pcd.points) == 0:
        logger.warning(f"No point cloud data for room {room_id}")
        return

    # Create visualization copy
    pcd_vis = room_pcd.__copy__()
    # Save if requested
    if save_path:
        save_visualization([pcd_vis], save_path, save_format)

    logger.info(f"Visualizing room {room_id} with {len(room_pcd.points)} points")
    o3d.visualization.draw_geometries(
        [pcd_vis], window_name=window_name, width=800, height=600
    )


def visualize_floor_pcd(
    floor_pcd: o3d.geometry.PointCloud,
    floor_id: str,
    window_name: Optional[str] = None,
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize a floor's point cloud.

    Args:
        floor_pcd: Point cloud of the floor
        floor_id: Floor identifier
        window_name: Optional window name
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if window_name is None:
        window_name = f"Floor {floor_id}"

    if floor_pcd is None or len(floor_pcd.points) == 0:
        logger.warning(f"No point cloud data for floor {floor_id}")
        return

    # Create visualization copy
    pcd_vis = floor_pcd.__copy__()
    pcd_vis.paint_uniform_color([0.6, 0.4, 0.2])  # Brown for floors

    # Save if requested
    if save_path:
        save_visualization([pcd_vis], save_path, save_format)

    logger.info(f"Visualizing floor {floor_id} with {len(floor_pcd.points)} points")
    o3d.visualization.draw_geometries(
        [pcd_vis], window_name=window_name, width=800, height=600
    )


def visualize_scene_hierarchy(
    floors: List[Any],
    rooms_by_floor: List[Tuple[Any, List[Any]]],
    nodes_by_room: Dict[str, List[Any]],  # Dict[str, List[ObjNode]]
    window_name: str = "Scene Hierarchy",
    save_path: Optional[str] = None,
    save_format: str = "ply",
) -> None:
    """
    Visualize the entire scene hierarchy: floors, rooms, and nodes.

    Args:
        floors: List of floor objects
        rooms_by_floor: List of (floor, rooms) tuples
        nodes_by_room: Dictionary mapping room_id to list of nodes
        window_name: Window name for visualization
        save_path: Optional path to save visualization (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    geometries = []

    # Generate colors for different hierarchical levels
    floor_colors = generate_colors(len(floors))

    for floor_idx, (floor, rooms) in enumerate(rooms_by_floor):
        floor_color = (
            floor_colors[floor_idx]
            if floor_idx < len(floor_colors)
            else [0.5, 0.5, 0.5]
        )

        for room_idx, room in enumerate(rooms):
            room_id = getattr(room, "room_id", f"room_{room_idx}")

            # Room point cloud (if available)
            if hasattr(room, "pcd") and room.pcd is not None:
                room_pcd = room.pcd.__copy__()
                # Lighter version of floor color for rooms
                room_color = floor_color * 0.7 + np.array([0.3, 0.3, 0.3])
                room_pcd.paint_uniform_color(room_color)
                geometries.append(room_pcd)

            # Node point clouds
            if room_id in nodes_by_room:
                node_colors = generate_colors(len(nodes_by_room[room_id]))
                for node_idx, node in enumerate(nodes_by_room[room_id]):
                    if node.pcd is not None and len(node.pcd.points) > 0:
                        node_pcd = o3d.geometry.PointCloud(node.pcd)
                        node_color = (
                            node_colors[node_idx]
                            if node_idx < len(node_colors)
                            else [0.5, 0.5, 0.5]
                        )
                        node_pcd.paint_uniform_color(node_color)
                        geometries.append(node_pcd)

    # Save if requested
    if save_path and geometries:
        save_visualization(geometries, save_path, save_format)

    if not geometries:
        logger.warning("No geometries to visualize in scene hierarchy")
        return

    logger.info(f"Visualizing scene hierarchy with {len(geometries)} geometries")
    o3d.visualization.draw_geometries(
        geometries, window_name=window_name, width=1400, height=900
    )


def visualize_2d_masks(
    masks: List[np.ndarray],
    rgb_frames: List[np.ndarray],
    node_id: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize 2D masks overlaid on RGB frames.

    Args:
        masks: List of 2D mask arrays
        rgb_frames: List of RGB frame arrays
        node_id: Node identifier for the window title
        save_path: Optional path to save the visualization
    """
    if not masks or not rgb_frames:
        return

    for i, (mask, rgb_frame) in enumerate(zip(masks, rgb_frames)):
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_frame)
        plt.title(f"RGB Frame {i}")
        plt.axis("off")

        # Mask overlay
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_frame)

        # Create colored mask overlay
        if mask.ndim == 2:
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = [1, 0, 0, 0.5]  # Red with transparency
            plt.imshow(colored_mask)

        plt.title(f"Mask Overlay {i}")
        plt.axis("off")

        plt.suptitle(f"Node {node_id} - Frame {i}")

        if save_path:
            output_path = f"{save_path}_2d_mask_{i}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved 2D visualization to {output_path}")

        plt.show()


def save_visualization(
    geometries: List[o3d.geometry.Geometry], save_path: str, save_format: str = "ply"
) -> None:
    """
    Save visualization geometries to file.

    Args:
        geometries: List of Open3D geometries to save
        save_path: Path to save (without extension)
        save_format: Format to save ('ply', 'pcd', 'obj')
    """
    if not geometries:
        logger.warning("No geometries to save")
        return

    save_format = save_format.lower()

    # Combine all point clouds if multiple exist
    if len(geometries) == 1:
        geometry = geometries[0]
    else:
        # Combine multiple point clouds
        combined_pcd = o3d.geometry.PointCloud()
        for geom in geometries:
            if isinstance(geom, o3d.geometry.PointCloud):
                combined_pcd += geom
        geometry = combined_pcd

    # Add appropriate extension
    if save_format == "ply":
        file_path = f"{save_path}.ply"
        success = o3d.io.write_point_cloud(file_path, geometry)
    elif save_format == "pcd":
        file_path = f"{save_path}.pcd"
        success = o3d.io.write_point_cloud(file_path, geometry)
    elif save_format == "obj":
        file_path = f"{save_path}.obj"
        # For OBJ, we might need to create a mesh first
        if isinstance(geometry, o3d.geometry.PointCloud):
            # Create a simple mesh from point cloud using ball pivoting
            try:
                mesh = create_mesh_from_pointcloud(geometry)
                success = o3d.io.write_triangle_mesh(file_path, mesh)
            except Exception as e:
                logger.warning(f"Failed to create mesh, saving as point cloud: {e}")
                file_path = f"{save_path}.ply"
                success = o3d.io.write_point_cloud(file_path, geometry)
        else:
            success = o3d.io.write_triangle_mesh(file_path, geometry)
    else:
        logger.error(f"Unsupported save format: {save_format}")
        return

    if success:
        logger.info(f"Saved visualization to {file_path}")
    else:
        logger.error(f"Failed to save visualization to {file_path}")


def create_mesh_from_pointcloud(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """
    Create a triangle mesh from a point cloud using Poisson reconstruction.

    Args:
        pcd: Input point cloud

    Returns:
        Triangle mesh
    """
    # Estimate normals
    pcd.estimate_normals()

    # Poisson reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Remove outlier vertices
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh


def create_visualization_summary(
    output_dir: str,
    nodes_by_room: Dict[str, List[Any]],  # Dict[str, List[ObjNode]]
    save_format: str = "ply",
) -> None:
    """
    Create a comprehensive visualization summary for all rooms and nodes.

    Args:
        output_dir: Directory to save visualizations
        nodes_by_room: Dictionary mapping room_id to list of nodes
        save_format: Format to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Creating visualization summary in {output_dir}")

    # Visualize each room separately
    for room_id, nodes in nodes_by_room.items():
        if not nodes:
            continue

        room_vis_path = os.path.join(output_dir, f"room_{room_id}_nodes")
        visualize_nodes_collection(
            nodes,
            window_name=f"Room {room_id} Nodes",
            save_path=room_vis_path,
            save_format=save_format,
        )

        # Functional elements for this room
        func_vis_path = os.path.join(output_dir, f"room_{room_id}_functional")
        visualize_functional_elements(
            nodes,
            window_name=f"Room {room_id} Functional Elements",
            save_path=func_vis_path,
            save_format=save_format,
        )

    # Create overall scene visualization
    all_nodes = []
    for nodes in nodes_by_room.values():
        all_nodes.extend(nodes)

    if all_nodes:
        scene_vis_path = os.path.join(output_dir, "scene_all_nodes")
        visualize_nodes_collection(
            all_nodes,
            window_name="All Scene Nodes",
            save_path=scene_vis_path,
            save_format=save_format,
        )

    logger.info(f"Visualization summary complete in {output_dir}")


def label_keyframe(
    rgb: np.ndarray,
    pose: np.ndarray,
    objects: List[Any],
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Draw 2D bounding boxes with object IDs on an RGB keyframe image.

    Projects each object's 3D point cloud into the camera frame and draws
    its 2D bounding box with the object label/ID.

    Args:
        rgb: RGB image (H, W, 3).
        pose: 4x4 camera-to-world transform.
        objects: List of ObjNode (must have .pcd, .id, .label).
        intrinsics: 3x3 camera intrinsic matrix matching the RGB resolution.
    Returns:
        Copy of the image with bounding boxes and labels drawn.
    """
    img = rgb.copy()
    h, w = img.shape[:2]
    world_to_cam = np.linalg.inv(pose)

    # Assign a unique color for each object
    colors = (generate_colors(len(objects)) * 255).astype(np.uint8)

    for idx, obj in enumerate(objects):
        if obj.pcd is None or len(obj.pcd.points) == 0:
            continue

        # Transform world points to camera frame
        pts = np.asarray(obj.pcd.points)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (world_to_cam @ pts_h.T).T[:, :3]

        # Keep only points in front of the camera
        front = pts_cam[:, 2] > 0.1
        if not np.any(front):
            continue
        pts_cam = pts_cam[front]

        # Project to pixel coordinates
        proj = (intrinsics @ pts_cam.T).T
        px = proj[:, 0] / proj[:, 2]
        py = proj[:, 1] / proj[:, 2]

        # Keep only points inside the image
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if np.sum(inside) < 3:
            continue
        px, py = px[inside], py[inside]

        # Remove masks smaller than .2% of the image resolution
        min_pixels = int(0.0005 * h * w)
        if len(px) < min_pixels:
            continue

        # Compute 2D bounding box
        x1, y1 = int(px.min()), int(py.min())
        x2, y2 = int(px.max()), int(py.max())

        # Skip tiny boxes
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        # color = tuple(int(c) for c in colors[idx])
        color = (255, 255, 0)
        # Limit label text to first two commas
        label_raw = str(getattr(obj, "label", ""))
        label_parts = label_raw.split(",")
        label_text = ",".join(label_parts[:2]).strip()
        id_text = str(obj.id)

        font_scale = 0.5
        thickness = 1

        # Write ID on top, label below
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        (idw, idh), _ = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        (lw, lh), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background rectangle for both lines, with extra space between ID and label
        gap = 10  # pixels of vertical space between ID and label
        rect_w = max(idw, lw)
        rect_h = idh + lh + gap + 6
        rect_x1 = cx - rect_w // 2
        rect_y1 = cy - rect_h // 2
        rect_x2 = cx + rect_w // 2
        rect_y2 = cy + rect_h // 2

        cv2.rectangle(
            img,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            color,
            -1,
        )
        # Draw ID (top)
        id_y = cy - rect_h // 2 + idh + 2
        cv2.putText(
            img,
            id_text,
            (cx - idw // 2, id_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        # Draw label (bottom, with gap)
        label_y = id_y + gap + lh // 2
        cv2.putText(
            img,
            label_text,
            (cx - lw // 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return img


def project_objects_to_masks(
    objects: List[Any],
    pose: np.ndarray,
    intrinsics: np.ndarray,
    h: int,
    w: int,
    min_pixels: int = 20,
) -> List[Tuple[Any, np.ndarray]]:
    """Project 3D object point clouds into 2D binary masks.

    Returns list of (object, mask) for objects visible in this view.
    """
    world_to_cam = np.linalg.inv(pose)
    visible = []

    for obj in objects:
        if obj.pcd is None or len(obj.pcd.points) == 0:
            continue

        pts = np.asarray(obj.pcd.points)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (world_to_cam @ pts_h.T).T[:, :3]

        front = pts_cam[:, 2] > 0.1
        if not np.any(front):
            continue
        pts_cam = pts_cam[front]

        proj = (intrinsics @ pts_cam.T).T
        px = (proj[:, 0] / proj[:, 2]).astype(int)
        py = (proj[:, 1] / proj[:, 2]).astype(int)

        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if np.sum(inside) < min_pixels:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[py[inside], px[inside]] = True
        visible.append((obj, mask))

    return visible


def match_detections_to_objects(
    det_masks: np.ndarray,
    obj_masks: List[Tuple[Any, np.ndarray]],
) -> List[Tuple[int, Any, float]]:
    """Match each detected 2D mask to the best-overlapping projected object.

    Args:
        det_masks: (N, H, W) boolean masks from the detector.
        obj_masks: List of (object, projected_mask) from project_objects_to_masks.

    Returns:
        List of (det_index, matched_object, iou) for matches with iou > 0.
    """
    matches = []
    for det_idx in range(len(det_masks)):
        det = det_masks[det_idx]
        best_iou, best_obj = 0.0, None
        for obj, proj_mask in obj_masks:
            intersection = np.logical_and(det, proj_mask).sum()
            union = np.logical_or(det, proj_mask).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_obj = obj
        if best_obj is not None and best_iou > 0.025:  # IoU threshold for a valid match
            matches.append((det_idx, best_obj, best_iou))
    return matches


def draw_id_labels(
    img: np.ndarray,
    det_masks: np.ndarray,
    matches: List[Tuple[int, Any, float]],
) -> np.ndarray:
    """Draw only object ID and label text at the center of each matched mask.

    No bounding boxes or mask overlays are drawn — just readable text for VLMs.
    """
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.5, 1

    for det_idx, obj, _ in matches:
        mask = det_masks[det_idx]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())

        id_text = str(obj.id)
        label_parts = str(getattr(obj, "label", "")).split(",")
        label_text = ",".join(label_parts[:2]).strip()

        (idw, idh), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
        (lw, lh), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        gap = 8
        rw = max(idw, lw) + 6
        rh = idh + lh + gap + 6
        rx1, ry1 = cx - rw // 2, cy - rh // 2
        rx2, ry2 = cx + rw // 2, cy + rh // 2

        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 255, 0), -1)
        cv2.putText(
            out,
            id_text,
            (cx - idw // 2, ry1 + idh + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        cv2.putText(
            out,
            label_text,
            (cx - lw // 2, ry1 + idh + gap + lh + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        # we could also draw the mask contours:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (0, 255, 255), 1)

    return out
