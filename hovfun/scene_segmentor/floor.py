"""Floor data class for scene segmentation."""

from __future__ import annotations
import os
import pickle
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import open3d as o3d

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .room import Room


class Floor:
    """
    Represents a floor in a building.

    Attributes:
        floor_id: Unique identifier for the floor
        name: Human-readable name
        rooms: List of Room objects on this floor
        pcd: Point cloud of the floor
        vertices: Bounding box vertices (8 points)
        floor_height: Height of the floor in meters
        floor_zero_level: Y-coordinate of the floor base
    """

    def __init__(self, floor_id: str, name: Optional[str] = None):
        self.floor_id = floor_id
        self.name = name or f"floor_{floor_id}"
        self.rooms: List[Room] = []
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.vertices: np.ndarray = np.array([])
        self.floor_height: Optional[float] = None
        self.floor_zero_level: Optional[float] = None

    def add_room(self, room: Room) -> None:
        """Add a room to this floor."""
        self.rooms.append(room)
        logger.debug(f"Added room {room.id} to floor {self.floor_id}")

    def save(self, path: str) -> None:
        """Save floor data to disk."""
        os.makedirs(path, exist_ok=True)

        if self.pcd is not None:
            pcd_path = os.path.join(path, f"{self.floor_id}.pcd")
            o3d.io.write_point_cloud(pcd_path, self.pcd)

        metadata = {
            "floor_id": self.floor_id,
            "name": self.name,
            "rooms": [r.id for r in self.rooms],
            "vertices": self.vertices.tolist() if hasattr(self.vertices, "tolist") else self.vertices,
            "floor_height": self.floor_height,
            "floor_zero_level": self.floor_zero_level,
        }
        pkl_path = os.path.join(path, f"{self.floor_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load floor data from disk."""
        pcd_path = os.path.join(path, f"{self.floor_id}.pcd")
        if os.path.exists(pcd_path):
            self.pcd = o3d.io.read_point_cloud(pcd_path)

        pkl_path = os.path.join(path, f"{self.floor_id}.pkl")
        with open(pkl_path, "rb") as f:
            metadata = pickle.load(f)
            self.name = metadata.get("name", self.name)
            self.vertices = np.asarray(metadata.get("vertices", []))
            self.floor_height = metadata.get("floor_height")
            self.floor_zero_level = metadata.get("floor_zero_level")

    @classmethod
    def load_from_file(cls, path: str, floor_id: str) -> Floor:
        """Load a floor from disk."""
        floor = cls(floor_id)
        floor.load(path)
        return floor

    def __str__(self) -> str:
        return f"Floor(id={self.floor_id}, name={self.name}, rooms={len(self.rooms)})"

    def __repr__(self) -> str:
        return self.__str__()
