"""Scene loading utilities for KeySG outputs.

Canonical implementation for loading floors, rooms, and objects from
pipeline output directories.
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
from loguru import logger
import pickle
import open3d as o3d

from keysg.scene_segmentor import Floor, Room
from keysg.scene_segmentor.obj_node import ObjNode as Object


def load_scene_nodes(output_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Load all nodes from a KeySG scene output directory.

    Scans: <output_dir>/segmentation/floor_*/room_*/nodes/*.pkl

    Returns:
        A nested dict: { floor_id: { room_id: [node_dict, ...], ... }, ... }
        Each node is the pickled dictionary saved during processing.
        Functional elements tagged as {"__type__": "ObjNode", "data": {...}}
        are inlined to just their "data" dict recursively for convenience.
    """

    def _inline_functional_elements(obj: Any) -> Any:
        # Recursively replace {"__type__":"ObjNode","data":{...}} with {...}
        if isinstance(obj, dict):
            if obj.get("__type__") == "ObjNode" and "data" in obj:
                return _inline_functional_elements(obj["data"])
            return {k: _inline_functional_elements(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_inline_functional_elements(v) for v in obj]
        return obj

    result: Dict[str, Dict[str, List[dict]]] = {}

    seg_dir = os.path.join(output_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        return result

    for floor_name in os.listdir(seg_dir):
        if not floor_name.startswith("floor_"):
            continue
        floor_dir = os.path.join(seg_dir, floor_name)
        if not os.path.isdir(floor_dir):
            continue
        floor_id = floor_name.replace("floor_", "", 1)
        result.setdefault(floor_id, {})

        for room_name in os.listdir(floor_dir):
            if not room_name.startswith("room_"):
                continue
            room_dir = os.path.join(floor_dir, room_name)
            nodes_dir = os.path.join(room_dir, "nodes")
            if not os.path.isdir(nodes_dir):
                continue
            room_id = room_name.replace("room_", "", 1)
            room_nodes: List[dict] = []

            for fname in os.listdir(nodes_dir):
                if not fname.endswith(".pkl"):
                    continue
                fpath = os.path.join(nodes_dir, fname)
                try:
                    if os.path.getsize(fpath) == 0:
                        continue
                    with open(fpath, "rb") as f:
                        node_dict = pickle.load(f)
                    room_nodes.append(_inline_functional_elements(node_dict))
                except Exception:
                    # Skip corrupted or unreadable files
                    continue

            if room_nodes:
                result[floor_id][room_id] = room_nodes
    # report number of loaded nodes
    num_floors = len(result)
    num_rooms = sum(len(rooms) for rooms in result.values())
    num_nodes = sum(len(nodes) for rooms in result.values() for nodes in rooms.values())
    print(
        f"Loaded {num_nodes} nodes from {num_rooms} rooms across {num_floors} floors."
    )
    return result


def get_objects(nodes: Dict[str, Dict[str, List[dict]]]) -> List[Object]:
    """Reconstruct ObjNode objects from nested dict structure produced by load_scene_nodes."""
    objects: List[Object] = []
    for floor_id, rooms in nodes.items():
        for room_id, room_nodes in rooms.items():
            for obj_dict in room_nodes:
                try:
                    if isinstance(obj_dict, dict):
                        obj_node = Object.from_dict(obj_dict)
                        objects.append(obj_node)
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct ObjNode for {floor_id}/{room_id}: {e}"
                    )
                    continue
    logger.info(f"Loaded {len(objects)} objects from scene.")
    return objects


def get_rooms(output_dir: str) -> Dict[str, Room]:
    """Load all Room objects from a KeySG scene output directory.

    Scans: <output_dir>/segmentation/floor_*/room_*/room.pkl

    Returns:
        A dict: { room_id: Room, ... }
    """
    rooms: Dict[str, Room] = {}
    seg_dir = os.path.join(output_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        return rooms

    for floor_name in os.listdir(seg_dir):
        if not floor_name.startswith("floor_"):
            continue
        floor_dir = os.path.join(seg_dir, floor_name)
        if not os.path.isdir(floor_dir):
            continue

        for room_name in os.listdir(floor_dir):
            if not room_name.startswith("room_"):
                continue
            room_dir = os.path.join(floor_dir, room_name)
            room_id = room_name.replace("room_", "", 1)
            room_pkl = os.path.join(room_dir, f"{room_id}.pkl")
            if not os.path.isfile(room_pkl):
                continue
            try:
                with open(room_pkl, "rb") as f:
                    loaded = pickle.load(f)

                if isinstance(loaded, dict):
                    # Minimal reconstruction; full Room object requires more info
                    room_obj = Room(
                        room_id, floor_id=floor_name.replace("floor_", "", 1)
                    )
                    room_obj.name = loaded.get("name", f"room_{room_id}")
                    room_obj.objects = loaded.get("objects", [])
                    room_obj.category = loaded.get("category")
                elif isinstance(loaded, Room):
                    room_obj = loaded
                else:
                    logger.warning(
                        f"Unrecognized room pickle type for {room_id}: {type(loaded)}"
                    )
                    continue

                pcd_path = os.path.join(room_dir, f"{room_id}.pcd")
                if os.path.exists(pcd_path):
                    try:
                        room_obj.pcd = o3d.io.read_point_cloud(pcd_path)
                    except Exception:
                        pass
                rooms[room_id] = room_obj

                # read .json if exists
                json_path = os.path.join(room_dir, f"room_{room_id}_vlm.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r") as f:
                            room_obj.vlm_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to read VLM data for {room_id}: {e}")
            except Exception as e:
                logger.error(f"Failed to load room {room_id}: {e}")
                continue
    logger.info(f"Loaded {len(rooms)} rooms from scene.")
    return rooms


def get_floors(output_dir: str) -> Dict[str, Floor]:
    """Load all Floor objects from a KeySG scene output directory.

    Scans: <output_dir>/segmentation/floor_*/floor.pkl

    Returns:
        A dict: { floor_id: Floor, ... }
    """
    floors: Dict[str, Floor] = {}
    seg_dir = os.path.join(output_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        return floors

    for floor_name in os.listdir(seg_dir):
        if not floor_name.startswith("floor_"):
            continue
        floor_dir = os.path.join(seg_dir, floor_name)
        if not os.path.isdir(floor_dir):
            continue
        floor_id = floor_name.replace("floor_", "", 1)
        logger.debug(f"Loading floor {floor_id}...")
        floor_pkl = os.path.join(floor_dir, f"{floor_id}.pkl")
        if not os.path.isfile(floor_pkl):
            continue
        try:
            with open(floor_pkl, "rb") as f:
                loaded = pickle.load(f)

            # If the pickle stored only metadata dict (as in Floor.save), rebuild Floor object
            if isinstance(loaded, dict):
                floor_obj = Floor(
                    floor_id, name=loaded.get("name", f"floor_{floor_id}")
                )
                floor_obj.rooms = loaded.get("rooms", [])
                verts = loaded.get("vertices", [])
                try:
                    floor_obj.vertices = np.asarray(verts)
                except Exception:
                    floor_obj.vertices = verts
                floor_obj.floor_height = loaded.get("floor_height")
                floor_obj.floor_zero_level = loaded.get("floor_zero_level")
            elif isinstance(loaded, Floor):
                floor_obj = loaded
            else:
                logger.warning(
                    f"Unrecognized floor pickle type for {floor_id}: {type(loaded)}"
                )
                continue

            # Load point cloud if available
            pcd_path = os.path.join(floor_dir, f"{floor_id}.pcd")
            if os.path.exists(pcd_path):
                try:
                    floor_obj.pcd = o3d.io.read_point_cloud(pcd_path)
                except Exception as e:
                    logger.warning(f"Failed reading floor pcd for {floor_id}: {e}")

            floors[floor_id] = floor_obj
        except Exception as e:
            logger.error(f"Failed to load floor {floor_id}: {e}")
            continue
    logger.info(f"Loaded {len(floors)} floors from scene.")
    return floors
