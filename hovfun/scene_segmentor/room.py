"""Room data class for scene segmentation."""

from __future__ import annotations
import os
import pickle
from typing import Any, List, Optional

import open3d as o3d
from shapely.geometry import Polygon

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class Room:
    """
    Represents a room within a floor.

    Attributes:
        id: Unique identifier for the room
        floor_id: ID of the parent floor
        name: Human-readable name
        objects: List of detected objects in this room
        polygon: 2D footprint polygon (XZ plane)
        pcd: 3D point cloud of the room
        height: Room height in meters
        zero_level: Y-coordinate of the room floor
        indices: Dataset frame indices assigned to this room
        sparse_indices: Representative frame indices (keyframes)
    """

    def __init__(self, id: str, floor_id: str, name: Optional[str] = None):
        self.id = id
        self.floor_id = floor_id
        self.name = name or f"room_{id}"
        self.category: Optional[str] = None
        self.objects: List[Any] = []
        self.polygon: Optional[Polygon] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.height: Optional[float] = None
        self.zero_level: Optional[float] = None
        self.indices: List[int] = []
        self.sparse_indices: List[int] = []

    def add_object(self, obj: Any) -> None:
        """Add an object to this room."""
        self.objects.append(obj)
        obj_id = getattr(obj, "id", "unknown")
        logger.debug(f"Added object {obj_id} to room {self.id}")

    def save(self, path: str) -> None:
        """Save room data to disk."""
        os.makedirs(path, exist_ok=True)

        # Save point cloud separately
        pcd_backup = self.pcd
        self.pcd = None

        if pcd_backup is not None and len(pcd_backup.points) > 0:
            pcd_path = os.path.join(path, f"{self.id}.pcd")
            o3d.io.write_point_cloud(pcd_path, pcd_backup)

        # Save metadata as pickle
        pkl_path = os.path.join(path, f"{self.id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        self.pcd = pcd_backup

    def load(self, path: str) -> None:
        """Load room data from disk."""
        pkl_path = os.path.join(path, f"{self.id}.pkl")
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
            for attr, value in loaded.__dict__.items():
                setattr(self, attr, value)

        pcd_path = os.path.join(path, f"{self.id}.pcd")
        if os.path.exists(pcd_path):
            self.pcd = o3d.io.read_point_cloud(pcd_path)

    @classmethod
    def load_from_file(cls, path: str, id: str) -> Room:
        """Load a room from disk."""
        pkl_path = os.path.join(path, f"{id}.pkl")
        with open(pkl_path, "rb") as f:
            room = pickle.load(f)

        pcd_path = os.path.join(path, f"{id}.pcd")
        if os.path.exists(pcd_path):
            room.pcd = o3d.io.read_point_cloud(pcd_path)

        return room

    def __str__(self) -> str:
        return (
            f"Room(id={self.id}, floor={self.floor_id}, "
            f"objects={len(self.objects)}, frames={len(self.indices)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
