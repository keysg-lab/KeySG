"""
SceneSegmentor: Main entry point for scene segmentation.

Responsibilities:
- Fuse RGB-D dataset into a point cloud
- Segment scene into floors and rooms
- Assign dataset frames to rooms and sample keyframes
- Save and load segmentation results
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from shapely.geometry import Point
from shapely import contains as shp_contains, points as shp_points

from .floor import Floor
from .floor_segmentation import FloorSegmentation
from .room import Room
from .room_segmentation import RoomSegmentation
from ..utils.frame_sampler import HDBSCANKeyframeSampler


class SceneSegmentor:
    """
    Segments a scene into floors and rooms from an RGB-D dataset.

    Usage:
        seg = SceneSegmentor(dataset, output_dir="output/pipeline")
        seg.run()
        seg.save()
    """

    def __init__(
        self,
        dataset: Any,
        output_dir: str,
        *,
        fuse_every_k: int = 10,
        voxel_size: float = 0.05,
        grid_resolution: float = 0.1,
        save_intermediate: bool = False,
        sampling_eps: float = 0.01,
        sampling_min_samples: int = 5,
        sampling_rot_weight: float = 1.5,
        points_in_room_threshold: float = 0.65,
        flip_zy: bool = False,
    ) -> None:
        self.dataset = dataset
        self.output_dir = output_dir
        self.fuse_every_k = fuse_every_k
        self.voxel_size = voxel_size
        self.grid_resolution = grid_resolution
        self.save_intermediate = save_intermediate
        self.sampling_eps = sampling_eps
        self.sampling_min_samples = sampling_min_samples
        self.sampling_rot_weight = sampling_rot_weight
        self.points_in_room_threshold = points_in_room_threshold
        self.flip_zy = flip_zy

        os.makedirs(self.output_dir, exist_ok=True)

        self._floors: List[Floor] = []
        self._floor_rooms: List[Tuple[Floor, List[Room]]] = []
        self._room_pose_dense: Dict[str, List[int]] = {}
        self._room_pose_sampled: Dict[str, List[int]] = {}

    def run(self) -> Tuple[List[Floor], List[Tuple[Floor, List[Room]]]]:
        """Execute the segmentation pipeline."""
        logger.info("Fusing scene point cloud...")
        scene_pcd = self._fuse_point_cloud()

        # Skip segmentation for non-HMP3D datasets
        skip_segmentation = getattr(self.dataset, "name", None) not in [
            "HMP3D",
            "AzureRGBD",
        ]

        if skip_segmentation:
            logger.info("Creating single floor/room (non-HMP3D dataset)")
            floors, floor_rooms = self._create_single_floor_room(scene_pcd)
        else:
            floors = self._segment_floors(scene_pcd)
            floor_rooms = self._segment_rooms(floors)

        self._assign_poses_to_rooms(floor_rooms)
        self._sample_keyframes(floor_rooms)

        self._floors = floors
        self._floor_rooms = floor_rooms
        return floors, floor_rooms

    def get_floors(self) -> List[Floor]:
        """Get detected floors."""
        return list(self._floors)

    def get_rooms_by_floor(self) -> List[Tuple[Floor, List[Room]]]:
        """Get (floor, rooms) pairs."""
        return self._floor_rooms

    def get_room_pose_indices(
        self,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Get (dense, sampled) pose indices per room."""
        return dict(self._room_pose_dense), dict(self._room_pose_sampled)

    def save(self, path: Optional[str] = None) -> str:
        """Save segmentation results to disk."""
        base = path or os.path.join(self.output_dir, "segmentation")
        os.makedirs(base, exist_ok=True)

        index = {"floors": [], "rooms": []}

        for floor, rooms in self._floor_rooms:
            f_dir = os.path.join(base, f"floor_{floor.floor_id}")
            os.makedirs(f_dir, exist_ok=True)
            floor.save(f_dir)
            index["floors"].append({"floor_id": floor.floor_id, "path": f_dir})

            for room in rooms:
                r_dir = os.path.join(f_dir, f"room_{room.id}")
                os.makedirs(r_dir, exist_ok=True)
                room.save(r_dir)
                index["rooms"].append(
                    {
                        "id": room.id,
                        "floor_id": floor.floor_id,
                        "path": r_dir,
                        "sparse_indices": room.sparse_indices,
                    }
                )
                self._save_keyframe_images(room, r_dir)

        index_path = os.path.join(base, "index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return index_path

    def load(
        self, path: Optional[str] = None
    ) -> Tuple[List[Floor], List[Tuple[Floor, List[Room]]]]:
        """Load segmentation results from disk."""
        base = path or os.path.join(self.output_dir, "segmentation")
        index_path = os.path.join(base, "index.json")

        with open(index_path) as f:
            index = json.load(f)

        # Load floors
        floor_by_id: Dict[str, Floor] = {}
        floors = []
        for fmeta in index.get("floors", []):
            floor = Floor.load_from_file(fmeta["path"], fmeta["floor_id"])
            floors.append(floor)
            floor_by_id[floor.floor_id] = floor

        # Load rooms
        rooms_by_floor: Dict[str, List[Room]] = {fid: [] for fid in floor_by_id}
        for rmeta in index.get("rooms", []):
            room = Room.load_from_file(rmeta["path"], rmeta["id"])
            rooms_by_floor[room.floor_id].append(room)

        floor_rooms = [
            (floor_by_id[fid], rooms) for fid, rooms in rooms_by_floor.items()
        ]

        # Restore pose assignments
        self._room_pose_dense = {
            r.id: r.indices for _, rooms in floor_rooms for r in rooms
        }
        self._room_pose_sampled = {
            r.id: r.sparse_indices for _, rooms in floor_rooms for r in rooms
        }

        self._floors = floors
        self._floor_rooms = floor_rooms

        for f, rs in floor_rooms:
            logger.info(f"Floor {f.floor_id}: {len(rs)} rooms")

        return floors, floor_rooms

    # ---- Private Methods ----

    def _fuse_point_cloud(self) -> o3d.geometry.PointCloud:
        """Fuse dataset frames into a single point cloud."""
        pcd = o3d.geometry.PointCloud()
        for i in tqdm(range(0, len(self.dataset), self.fuse_every_k), desc="Fusing"):
            rgb, depth, pose = self.dataset[i]
            frame_pcd = self.dataset.create_pcd(rgb, depth, pose)
            pcd += frame_pcd

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        logger.info(f"Fused point cloud: {len(pcd.points)} points")
        return pcd

    def _segment_floors(self, pcd: o3d.geometry.PointCloud) -> List[Floor]:
        """Segment point cloud into floors."""
        logger.info("Segmenting floors...")
        seg = FloorSegmentation(pcd, save_intermediate=self.save_intermediate)
        floors = seg.segment_floors(output_path=self.output_dir, flip_zy=self.flip_zy)
        logger.info(f"Found {len(floors)} floors")
        return floors

    def _segment_rooms(self, floors: List[Floor]) -> List[Tuple[Floor, List[Room]]]:
        """Segment each floor into rooms."""
        logger.info("Segmenting rooms...")
        room_seg = RoomSegmentation(save_intermediate=self.save_intermediate)
        floor_rooms = []

        for floor in floors:
            f_dir = os.path.join(self.output_dir, f"floor_{floor.floor_id}")
            os.makedirs(f_dir, exist_ok=True)
            rooms = room_seg.segment_rooms_from_floor(
                floor, f_dir, self.grid_resolution
            )
            logger.info(f"Floor {floor.floor_id}: {len(rooms)} rooms")
            floor_rooms.append((floor, rooms))

        return floor_rooms

    def _create_single_floor_room(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> Tuple[List[Floor], List[Tuple[Floor, List[Room]]]]:
        """Create a single floor and room for simple scenes."""
        floor = Floor("0", "floor_0")
        floor.pcd = pcd

        if len(pcd.points) > 0:
            pts = np.asarray(pcd.points)
            floor.vertices = np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()
            )
            floor.floor_zero_level = float(np.nanmin(pts[:, 1]))
            floor.floor_height = float(np.nanmax(pts[:, 1]) - floor.floor_zero_level)

        room = Room("0_0", floor.floor_id, "room_0")
        room.pcd = pcd
        room.zero_level = floor.floor_zero_level
        room.height = floor.floor_height
        floor.add_room(room)

        return [floor], [(floor, [room])]

    def _assign_poses_to_rooms(
        self, floor_rooms: List[Tuple[Floor, List[Room]]]
    ) -> None:
        """Assign dataset frame indices to rooms based on camera position."""
        logger.info("Assigning poses to rooms...")

        # Precompute poses and floor bounds
        poses = [self.dataset[i][2] for i in range(len(self.dataset))]
        floor_bounds = self._compute_floor_bounds(floor_rooms)

        # Initialize room indices
        for _, rooms in floor_rooms:
            for r in rooms:
                r.indices = []

        # Assign each pose to a room
        for idx, pose in tqdm(enumerate(poses), total=len(poses), desc="Assigning"):
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]

            for floor, rooms in floor_rooms:
                y_min, y_max = floor_bounds[floor.floor_id]
                if not (y_min - 0.5 <= y <= y_max + 0.5):
                    continue

                # Fast path: single room per floor
                if len(rooms) == 1:
                    rooms[0].indices.append(idx)
                    break

                # Check room containment
                for room in rooms:
                    if room.polygon is None:
                        continue
                    if room.polygon.contains(Point(x, z)):
                        if self._validate_pose_in_room(idx, room):
                            room.indices.append(idx)
                            break
                break

        # Store dense assignments
        self._room_pose_dense = {
            r.id: r.indices for _, rooms in floor_rooms for r in rooms
        }

    def _sample_keyframes(self, floor_rooms: List[Tuple[Floor, List[Room]]]) -> None:
        """Sample representative keyframes for each room."""
        logger.info("Sampling keyframes...")

        for _, rooms in floor_rooms:
            for room in rooms:
                if not room.indices:
                    room.sparse_indices = []
                    continue
                sampler = HDBSCANKeyframeSampler(
                    self.dataset, selected_indices=room.indices
                )
                sampled = sampler.sample_hdbscan(
                    min_cluster_size=15,
                )
                room.sparse_indices = sampled if sampled else list(room.indices)
                logger.info(
                    f"Room {room.id}: {len(room.sparse_indices)} keyframes from {len(room.indices)} frames"
                )

        self._room_pose_sampled = {
            r.id: r.sparse_indices for _, rooms in floor_rooms for r in rooms
        }

    def _compute_floor_bounds(
        self,
        floor_rooms: List[Tuple[Floor, List[Room]]],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute vertical bounds for each floor."""
        bounds = {}
        for floor, _ in floor_rooms:
            if floor.floor_zero_level is not None and floor.floor_height is not None:
                bounds[floor.floor_id] = (
                    float(floor.floor_zero_level),
                    float(floor.floor_zero_level + floor.floor_height),
                )
            else:
                pts = np.asarray(floor.pcd.points)
                y_min, y_max = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])
                bounds[floor.floor_id] = (y_min, y_max)
        return bounds

    def _validate_pose_in_room(
        self, idx: int, room: Room, threshold: float = 0.5
    ) -> bool:
        """Validate that a frame's point cloud mostly falls within the room."""
        if room.polygon is None:
            return False

        try:
            rgb, depth, pose = self.dataset[idx]
            frame_pcd = self.dataset.create_pcd(rgb, depth, pose).voxel_down_sample(0.1)
        except Exception:
            return False

        pts = np.asarray(frame_pcd.points)
        if len(pts) == 0:
            return False

        pts_2d = pts[:, [0, 2]]

        try:
            point_geoms = shp_points(pts_2d)
            inside = shp_contains(room.polygon, point_geoms)
            return float(np.sum(inside)) / len(pts_2d) >= threshold
        except Exception:
            return False

    def _save_keyframe_images(self, room: Room, output_dir: str) -> None:
        """Save keyframe images for a room."""
        img_dir = os.path.join(output_dir, "keyframes")
        os.makedirs(img_dir, exist_ok=True)

        for idx in room.sparse_indices:
            rgb, _, _ = self.dataset[idx]
            img_path = os.path.join(img_dir, f"frame_{idx:06d}.png")
            o3d.io.write_image(img_path, o3d.geometry.Image(rgb))
