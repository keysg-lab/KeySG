import numpy as np
import open3d as o3d
from dataclasses import dataclass


@dataclass
class ObjNode:
    """A node in the 3D scene graph."""

    id: str
    bbox_3d: list | None = None
    bboxs_2d: list | None = None
    label: str = ""
    pcd: o3d.geometry.PointCloud | None = None
    feature: np.ndarray | None = None
    text_feature: np.ndarray | None = None
    masks_2d: list | None = None
    rgb_frames: list | None = None  # Store RGB frames corresponding to masks_2d
    frame_indices: list | None = None  # Store frame indices corresponding to masks_2d
    functional_elements: list | None = None
    best_crop: np.ndarray | None = None  # tight bbox crop of best view (uint8 HxWx3)
    vlm_description: dict | None = None  # VLM output: name, description, attributes, …

    def to_dict(self):
        """
        Serialize ObjNode to a dictionary with only pickle-able types.
        """

        def pcd_to_dict(pcd):
            if pcd is None:
                return None
            return {
                "points": np.asarray(pcd.points),
                "colors": (np.asarray(pcd.colors) if pcd.has_colors() else None),
            }

        def functional_element_to_dict(elem):
            """Safely serialize a functional element that may be an ObjNode or any JSON/pickle-able object."""
            if isinstance(elem, ObjNode):
                # Tag nested ObjNode to disambiguate on load
                return {"__type__": "ObjNode", "data": elem.to_dict()}
            # Assume elem is already pickle-able (e.g., str, int, dict, list)
            return elem

        return {
            "id": self.id,
            "bbox_3d": self.bbox_3d,
            "bboxs_2d": self.bboxs_2d,
            "label": self.label,
            "pcd": pcd_to_dict(self.pcd),
            "feature": self.feature,
            "text_feature": self.text_feature,
            "masks_2d": self.masks_2d,
            "rgb_frames": self.rgb_frames,
            "frame_indices": self.frame_indices,
            "functional_elements": (
                [functional_element_to_dict(e) for e in self.functional_elements]
                if self.functional_elements is not None
                else None
            ),
            "best_crop": self.best_crop,
            "vlm_description": self.vlm_description,
        }

    @staticmethod
    def from_dict(d):
        """
        Deserialize ObjNode from a dictionary.
        """

        def dict_to_pcd(pcd_dict):
            if not pcd_dict:
                return None
            pcd = o3d.geometry.PointCloud()
            # Data is already a NumPy array, no np.array() cast needed
            pcd.points = o3d.utility.Vector3dVector(pcd_dict["points"])
            if pcd_dict.get("colors") is not None:
                pcd.colors = o3d.utility.Vector3dVector(pcd_dict["colors"])
            return pcd

        def functional_element_from_dict(elem):
            """Reconstruct functional elements, restoring nested ObjNode when tagged."""
            if isinstance(elem, dict) and elem.get("__type__") == "ObjNode":
                return ObjNode.from_dict(elem.get("data", {}))
            # Leave anything else as-is
            return elem

        return ObjNode(
            id=d["id"],
            bbox_3d=d.get("bbox_3d"),
            bboxs_2d=d.get("bboxs_2d"),
            label=d.get("label", ""),
            pcd=dict_to_pcd(d.get("pcd")),
            feature=d.get("feature"),
            text_feature=d.get("text_feature"),
            masks_2d=d.get("masks_2d"),
            rgb_frames=d.get("rgb_frames"),
            frame_indices=d.get("frame_indices"),
            functional_elements=(
                [
                    functional_element_from_dict(e)
                    for e in d.get("functional_elements", [])
                ]
                if d.get("functional_elements") is not None
                else None
            ),
            best_crop=d.get("best_crop"),
            vlm_description=d.get("vlm_description"),
        )
