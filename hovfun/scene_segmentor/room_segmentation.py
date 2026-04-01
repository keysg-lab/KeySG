"""Room segmentation algorithm using distance transform and watershed."""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .room import Room

if TYPE_CHECKING:
    from .floor import Floor


class RoomSegmentation:
    """
    Segments a floor point cloud into rooms using distance transform and watershed.

    Algorithm:
    1. Slice point cloud to wall-detection height
    2. Create 2D occupancy grid from XZ projection
    3. Extract wall skeleton and building boundary
    4. Apply distance transform to find room centers
    5. Use watershed to segment into rooms
    6. Create Room objects with 3D point clouds
    """

    # Slicing parameters (meters)
    WALL_SLICE_LOWER: float = 1.5
    WALL_SLICE_UPPER_MARGIN: float = 0.3
    FULL_SLICE_UPPER_MARGIN: float = 0.2

    # Morphology parameters
    BORDER_SIZE: int = 10
    MIN_ROOM_AREA_M2: float = 0.5

    def __init__(self, save_intermediate: bool = False):
        self.save_intermediate = save_intermediate
        self.rooms: List[Room] = []

    def segment_rooms_from_floor(
        self,
        floor: Floor,
        output_path: Optional[str] = None,
        grid_resolution: float = 0.1,
    ) -> List[Room]:
        """
        Segment rooms from a floor's point cloud.

        Args:
            floor: Floor object with point cloud
            output_path: Path for intermediate visualizations
            grid_resolution: Grid cell size in meters

        Returns:
            List of Room objects
        """
        logger.info(f"Segmenting rooms for floor {floor.floor_id}")

        if floor.pcd is None or len(floor.pcd.points) == 0:
            logger.warning(f"Floor {floor.floor_id} has no point cloud")
            return []

        # Prepare output directory
        floor_output = None
        if output_path:
            floor_output = os.path.join(output_path, f"floor_{floor.floor_id}")
            os.makedirs(floor_output, exist_ok=True)

        # Extract floor properties
        xyz = np.asarray(floor.pcd.points)
        y_base = floor.floor_zero_level
        y_height = floor.floor_height

        # Slice point cloud for wall detection
        wall_pts = self._slice_points(xyz, y_base + self.WALL_SLICE_LOWER,
                                      y_base + y_height - self.WALL_SLICE_UPPER_MARGIN)
        full_pts = self._slice_points(xyz, None,
                                      y_base + y_height - self.FULL_SLICE_UPPER_MARGIN)

        if len(wall_pts) == 0:
            logger.warning("No wall points after slicing")
            return []

        # Project to 2D (XZ plane)
        wall_2d = wall_pts[:, [0, 2]]
        full_2d = full_pts[:, [0, 2]]

        # Detect room regions
        room_data = self._detect_rooms(wall_2d, full_2d, grid_resolution, floor_output)

        if not room_data:
            logger.warning("No rooms detected")
            return []

        # Create Room objects
        self.rooms = self._create_rooms(room_data, floor, y_base, y_height)
        logger.info(f"Created {len(self.rooms)} rooms")
        return self.rooms

    def _slice_points(
        self,
        xyz: np.ndarray,
        y_min: Optional[float],
        y_max: Optional[float],
    ) -> np.ndarray:
        """Filter points by Y range."""
        mask = np.ones(len(xyz), dtype=bool)
        if y_min is not None:
            mask &= xyz[:, 1] >= y_min
        if y_max is not None:
            mask &= xyz[:, 1] < y_max
        return xyz[mask]

    def _detect_rooms(
        self,
        wall_2d: np.ndarray,
        full_2d: np.ndarray,
        resolution: float,
        output_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Detect room regions using distance transform."""
        # Compute grid bounds
        min_bounds = np.min(wall_2d, axis=0)
        max_bounds = np.max(wall_2d, axis=0)
        grid_size = max_bounds - min_bounds
        num_bins = (int(grid_size[1] / resolution), int(grid_size[0] / resolution))

        # Create wall skeleton
        walls = self._create_wall_skeleton(wall_2d, num_bins)

        # Create building boundary
        boundary = self._create_boundary(full_2d, num_bins)

        # Combine into occupancy map
        occupancy = cv2.bitwise_or(walls, cv2.bitwise_not(boundary))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.save_intermediate and output_path:
            self._save_maps(walls, boundary, occupancy, output_path)

        # Apply distance transform to find rooms
        polygons = self._distance_transform_rooms(occupancy, resolution, output_path)

        # Convert to world coordinates
        rooms = []
        for i, poly in enumerate(polygons):
            world_coords = self._pixel_to_world(poly, resolution, min_bounds)
            room_poly = Polygon(world_coords).buffer(0.3)
            if room_poly.is_valid and room_poly.area > 0:
                rooms.append({
                    "label": i + 1,
                    "polygon": room_poly,
                    "area": room_poly.area,
                })

        return rooms

    def _create_wall_skeleton(self, pts_2d: np.ndarray, num_bins: Tuple[int, int]) -> np.ndarray:
        """Create wall skeleton from 2D histogram."""
        hist, _, _ = np.histogram2d(pts_2d[:, 1], pts_2d[:, 0], bins=num_bins)
        hist_norm = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist_blur = cv2.GaussianBlur(hist_norm, (5, 5), 1.0)

        threshold = 0.25 * np.max(hist_blur)
        _, skeleton = cv2.threshold(hist_blur, threshold, 255, cv2.THRESH_BINARY)

        # Add border
        skeleton = cv2.copyMakeBorder(
            skeleton, self.BORDER_SIZE, self.BORDER_SIZE,
            self.BORDER_SIZE, self.BORDER_SIZE, cv2.BORDER_CONSTANT, value=0
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        return cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _create_boundary(self, pts_2d: np.ndarray, num_bins: Tuple[int, int]) -> np.ndarray:
        """Create building boundary from full point cloud."""
        hist, _, _ = np.histogram2d(pts_2d[:, 1], pts_2d[:, 0], bins=num_bins)
        hist_norm = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist_blur = cv2.GaussianBlur(hist_norm, (21, 21), 2.0)

        _, boundary = cv2.threshold(hist_blur, 0, 255, cv2.THRESH_BINARY)

        # Add border
        boundary = cv2.copyMakeBorder(
            boundary, self.BORDER_SIZE, self.BORDER_SIZE,
            self.BORDER_SIZE, self.BORDER_SIZE, cv2.BORDER_CONSTANT, value=0
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Keep only outer contour
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(boundary)
        cv2.drawContours(result, contours, -1, 255, -1)
        return result.astype(np.uint8)

    def _distance_transform_rooms(
        self,
        occupancy: np.ndarray,
        resolution: float,
        output_path: Optional[str],
    ) -> List[Polygon]:
        """Use distance transform and watershed to find room regions."""
        # Invert and compute distance transform
        inverted = cv2.bitwise_not(occupancy.astype(np.uint8))
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Find seed regions
        blur = cv2.GaussianBlur(dist_norm, (11, 11), 10.0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        min_area_px = (self.MIN_ROOM_AREA_M2 / resolution) ** 2
        seeds = [c for c in contours if cv2.contourArea(c) > min_area_px]
        logger.info(f"Found {len(seeds)} room seeds")

        if not seeds:
            return []

        # Watershed segmentation
        markers = np.zeros(occupancy.shape, dtype=np.int32)
        for i, contour in enumerate(seeds):
            cv2.drawContours(markers, [contour], -1, i + 1, -1)
        cv2.circle(markers, (3, 3), 1, len(seeds) + 1, -1)  # Background marker

        bgr = cv2.cvtColor(occupancy, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(bgr, markers)

        if self.save_intermediate and output_path:
            self._save_watershed(dist_norm, thresh, markers, output_path)

        # Extract polygons from markers
        polygons = []
        for i in range(1, len(seeds) + 1):
            mask = (markers == i).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                pts = cnts[0].squeeze()
                if len(pts) >= 3:
                    polygons.append(Polygon(pts))

        return polygons

    def _pixel_to_world(
        self,
        polygon: Polygon,
        resolution: float,
        origin: np.ndarray,
    ) -> np.ndarray:
        """Convert pixel polygon to world coordinates."""
        x, y = polygon.exterior.xy
        x, y = np.array(x), np.array(y)
        return np.column_stack([
            origin[0] + x * resolution - 1,
            origin[1] + y * resolution - 1,
        ])

    def _create_rooms(
        self,
        room_data: List[Dict[str, Any]],
        floor: Floor,
        y_base: float,
        y_height: float,
    ) -> List[Room]:
        """Create Room objects from detected regions."""
        rooms = []
        for i, data in enumerate(room_data):
            room = Room(f"{floor.floor_id}_{i}", floor.floor_id, f"room_{i}")
            room.polygon = data["polygon"]
            room.height = y_height
            room.zero_level = y_base

            # Extract 3D points within room polygon
            room.pcd = self._extract_room_points(floor, data["polygon"], y_base, y_height)

            if room.pcd is not None and len(room.pcd.points) > 0:
                floor.add_room(room)
                rooms.append(room)

        return rooms

    def _extract_room_points(
        self,
        floor: Floor,
        polygon: Polygon,
        y_base: float,
        y_height: float,
    ) -> Optional[o3d.geometry.PointCloud]:
        """Extract points within room polygon from floor point cloud."""
        if floor.pcd is None:
            return None

        xyz = np.asarray(floor.pcd.points)
        colors = np.asarray(floor.pcd.colors)

        # Filter by height
        eps = 0.1
        mask = (xyz[:, 1] >= y_base - eps) & (xyz[:, 1] <= y_base + y_height + eps)
        xyz_h = xyz[mask]
        colors_h = colors[mask]

        if len(xyz_h) == 0:
            return None

        # Filter by polygon containment
        xz = xyz_h[:, [0, 2]]
        try:
            poly_pts = np.array(polygon.exterior.coords)
            path = MplPath(poly_pts)
            inside = path.contains_points(xz, radius=0.1)
        except Exception:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_h[inside])
        pcd.colors = o3d.utility.Vector3dVector(colors_h[inside])
        return pcd

    def _save_maps(
        self,
        walls: np.ndarray,
        boundary: np.ndarray,
        occupancy: np.ndarray,
        output_path: str,
    ) -> None:
        """Save intermediate map visualizations."""
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(walls, cmap="gray", origin="lower")
            axes[0].set_title("Walls")
            axes[1].imshow(boundary, cmap="gray", origin="lower")
            axes[1].set_title("Boundary")
            axes[2].imshow(occupancy, cmap="gray", origin="lower")
            axes[2].set_title("Occupancy")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "room_maps.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save maps: {e}")

    def _save_watershed(
        self,
        dist: np.ndarray,
        thresh: np.ndarray,
        markers: np.ndarray,
        output_path: str,
    ) -> None:
        """Save watershed visualization."""
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(dist, cmap="jet", origin="lower")
            axes[0].set_title("Distance Transform")
            axes[1].imshow(thresh, cmap="gray", origin="lower")
            axes[1].set_title("Thresholded")
            axes[2].imshow(markers, cmap="tab20", origin="lower")
            axes[2].set_title("Watershed")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "watershed.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save watershed: {e}")
