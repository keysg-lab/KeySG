import os
import numpy as np
import cv2
import open3d as o3d
import json

class ScanNetDataset:
    """
    Dataset class for the ScanNet dataset, matching the interface of ReplicaDataset.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration dictionary containing dataset parameters.
        """
        self.root_dir = cfg.get("root_dir", "")
        self.depth_scale = cfg.get("depth_scale", 1000.0)
        self.depth_min = cfg.get("depth_min", 0.5)
        self.depth_max = cfg.get("depth_max", 4.0)
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")
        self.transforms = cfg.get("transforms", None)

        # Load intrinsics
        self.rgb_intrinsics = self._load_rgb_intrinsics(
            os.path.join(self.root_dir, "intrinsic/intrinsic_color.txt")
        )
        self.depth_intrinsics = self._load_depth_intrinsics(
            os.path.join(self.root_dir, "intrinsic/intrinsic_depth.txt")
        )
        self.data_list = self._get_data_list()
        self._gt_object_points = None
        self.gt_objects = (
            {}
        )  # Map: object_id -> {'centroid': (3,), 'label': str, 'segments': []}
        self._load_gt_metadata()
        
        # Get image shape
        sample_image = self._load_image(self.data_list[0][0])
        self.rgb_H = sample_image.shape[0]
        self.rgb_W = sample_image.shape[1]
        self.depth_H = self.depth_intrinsics.shape[0]
        self.depth_W = self.depth_intrinsics.shape[1]
        self.name = "ScanNet"
        self.scene_name = self.root_dir.split("/")[-1]

    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image, depth image, and pose as numpy arrays.
        """
        rgb_path, depth_path, pose_path = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(pose_path)
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
        return rgb_image, depth_image, pose

    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        rgb_data_list = os.listdir(os.path.join(self.root_dir, "color"))
        rgb_data_list = [os.path.join(self.root_dir, "color", x) for x in rgb_data_list]
        depth_data_list = os.listdir(os.path.join(self.root_dir, "depth"))
        depth_data_list = [
            os.path.join(self.root_dir, "depth", x) for x in depth_data_list
        ]
        pose_data_list = os.listdir(os.path.join(self.root_dir, "pose"))
        pose_data_list = [
            os.path.join(self.root_dir, "pose", x) for x in pose_data_list
        ]
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list.sort()
        return list(zip(rgb_data_list, depth_data_list, pose_data_list))

    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        rgb_image = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def _load_depth(self, path):
        """
        Load the depth image from the given path.

        Args:
            path: Path to the depth image file.

        Returns:
            Depth image as a numpy array.
        """
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return depth_image

    def _load_pose(self, path):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        with open(path, "r") as f:
            pose = []
            for line in f:
                pose.append([float(x) for x in line.split()])
        pose = np.array(pose)
        return pose

    def _load_rgb_intrinsics(self, path):
        """
        Load the RGB camera intrinsics from the given path.

        Args:
            path: Path to the RGB camera intrinsics file.

        Returns:
            RGB camera intrinsics as a numpy array (3x3 matrix).
        """
        with open(path, "r") as f:
            intrinsics = []
            for line in f:
                intrinsics.append([float(x) for x in line.split()])
        intrinsics = np.array(intrinsics)
        intrinsics = self._get_3x3_intrinsics(intrinsics)
        return intrinsics

    def _load_depth_intrinsics(self, path):
        """
        Load the depth camera intrinsics from the given path.

        Args:
            path: Path to the depth camera intrinsics file.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """
        with open(path, "r") as f:
            intrinsics = []
            for line in f:
                intrinsics.append([float(x) for x in line.split()])
        intrinsics = np.array(intrinsics)
        intrinsics = self._get_3x3_intrinsics(intrinsics)
        return intrinsics

    def _get_3x3_intrinsics(self, intrinsics: np.ndarray) -> np.ndarray:
        """Return a 3x3 camera matrix from input intrinsics.

        ScanNet intrinsics files are often 4x4 with the 3x3 K in the top-left.
        This utility extracts the 3x3 if needed.
        """
        if intrinsics.shape == (3, 3):
            return intrinsics
        if intrinsics.shape == (4, 4):
            return intrinsics[:3, :3]
        raise ValueError(f"Unsupported intrinsics shape: {intrinsics.shape}")
    
    def create_pcd(self, rgb, depth, camera_pose=None):
        """
        Create a point cloud from RGB-D images using Open3D's optimized functions.

        Args:
            rgb: RGB image as a numpy array.
            depth: Depth image as a numpy array.
            camera_pose: Camera pose as a numpy array (4x4 matrix).

        Returns:
            Point cloud as an Open3D object.
        """
        rgb = np.ascontiguousarray(rgb)
        depth = np.ascontiguousarray(depth)

        # 1. Align RGB and Depth resolutions
        h_d, w_d = depth.shape[:2]
        h_r, w_r = rgb.shape[:2]

        if h_d != h_r or w_d != w_r:
            rgb = cv2.resize(rgb, (w_d, h_d), interpolation=cv2.INTER_LINEAR)

        # 2. Filter Depth (Min/Max)
        min_depth_raw = self.depth_min * self.depth_scale
        max_depth_raw = self.depth_max * self.depth_scale
        
        # Zero out pixels outside the valid range (Open3D ignores 0 depth)
        mask = (depth < min_depth_raw) | (depth > max_depth_raw)
        depth = depth.copy()  # Avoid modifying the original array
        depth[mask] = 0

        # 3. Create Open3D structures
        o3d_color = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False
        )

        # 4. Construct Pinhole Intrinsic Object
        # Extract fx, fy, cx, cy from the stored 3x3 intrinsics matrix
        fx = self.depth_intrinsics[0, 0]
        fy = self.depth_intrinsics[1, 1]
        cx = self.depth_intrinsics[0, 2]
        cy = self.depth_intrinsics[1, 2]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w_d,
            height=h_d,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # 5. Generate Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )

        # 6. Transform
        if camera_pose is not None:
            pcd.transform(camera_pose)

        return pcd

    def project_2d_mask_to_3d(
        self,
        mask_2d: np.ndarray,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        camera_pose: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Project 2D mask to 3D point cloud using dataset's camera parameters with RGB colors.

        Args:
            mask_2d: 2D binary mask
            depth_image: Depth image
            rgb_image: RGB image for color information
            camera_pose: Camera pose (4x4 transformation matrix)

        Returns:
            Open3D point cloud object with RGB colors
        """
        Hd, Wd = depth_image.shape[:2]

        # Ensure mask and RGB are aligned to depth resolution
        if mask_2d.shape[:2] != (Hd, Wd):
            mask_resized = cv2.resize(
                mask_2d.astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_2d

        if rgb_image.shape[:2] != (Hd, Wd):
            rgb_resized = cv2.resize(
                rgb_image, (Wd, Hd), interpolation=cv2.INTER_LINEAR
            )
        else:
            rgb_resized = rgb_image

        y, x = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")

        # Apply mask
        mask_indices = mask_resized > 0
        x_masked = x[mask_indices]
        y_masked = y[mask_indices]

        # Get depth values
        depth = depth_image.astype(np.float32) / self.depth_scale
        depth_masked = depth[mask_indices]

        # Get RGB values for masked pixels
        rgb_masked = rgb_resized[mask_indices]

        # Filter by depth range
        valid_depth = np.logical_and(
            depth_masked > self.depth_min, depth_masked < self.depth_max
        )
        x_valid = x_masked[valid_depth]
        y_valid = y_masked[valid_depth]
        depth_valid = depth_masked[valid_depth]
        rgb_valid = rgb_masked[valid_depth]

        if len(x_valid) == 0:
            return o3d.geometry.PointCloud()

        # Convert to 3D coordinates using dataset's intrinsics
        pixels = np.stack([x_valid, y_valid, np.ones_like(x_valid)], axis=0)
        K = self._get_3x3_intrinsics(self.depth_intrinsics)
        K_inv = np.linalg.inv(K)
        xyz = K_inv @ pixels * depth_valid

        # Transform to world coordinates
        points_cam = np.vstack([xyz, np.ones((1, xyz.shape[1]))])
        points_world = camera_pose @ points_cam

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world[:3, :].T)
        pcd.colors = o3d.utility.Vector3dVector(rgb_valid / 255.0)

        return pcd

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.data_list)


    def _find_scannet_file(self, suffixes):
        for sfx in suffixes:
            match = next(
                (f for f in os.listdir(self.root_dir) if f.endswith(sfx)),
                None,
            )
            if match:
                return os.path.join(self.root_dir, match)
        return None
    
    def _load_gt_metadata(self):
        """
        Loads 3D object segments and centroids from ScanNet aggregation files.
        """
        # Look for the aggregation file (links segments to labels/objects)
        agg_file = next(
            (f for f in os.listdir(self.root_dir) if f.endswith(".aggregation.json")),
            None,
        )

        if not agg_file:
            print(
                f"[WARN] Aggregation file not found in {self.root_dir}. Searching for alternative GT files..."
            )
            # Fallback for specific naming conventions if standard fails
            agg_file = next(
                (
                    f
                    for f in os.listdir(self.root_dir)
                    if "vh_clean" in f and f.endswith(".json") and "segs" not in f
                ),
                None,
            )

        if not agg_file:
            print(
                "[ERROR] GT Metadata (aggregation.json) missing. Semantic metrics will be unavailable."
            )
            return

        try:
            with open(os.path.join(self.root_dir, agg_file), "r") as f:
                agg_data = json.load(f)

            for obj in agg_data.get("segGroups", []):
                # ScanNet provides OBB (Oriented Bounding Box) info which includes the centroid
                centroid = np.zeros(3)
                if "obb" in obj:
                    centroid = np.array(obj["obb"]["centroid"])
                elif "loc" in obj:  # Alternative metadata format
                    centroid = np.array(obj["loc"])

                self.gt_objects[obj["id"]] = {
                    "label": obj["label"],
                    "segments": obj["segments"],
                    "centroid": centroid,
                }
            print(f"[INFO] Loaded {len(self.gt_objects)} GT objects from {agg_file}")
        except Exception as e:
            print(f"[ERROR] Failed to parse GT metadata: {e}")

    def _list_scannet_mesh_candidates(self):
        """Return candidate mesh paths in preference order."""
        suffixes = [
            "_vh_clean_2.ply",
            "_vh_clean.ply",
            ".ply",
        ]
        candidates = []
        for sfx in suffixes:
            for f in os.listdir(self.root_dir):
                if f.endswith(sfx):
                    candidates.append(os.path.join(self.root_dir, f))
        # Preserve order while removing duplicates
        seen = set()
        ordered = []
        for p in candidates:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        return ordered

    def _select_mesh_matching_segs(self, seg_indices, mesh_path=None):
        """Select a mesh whose vertex count matches segIndices length."""
        candidates = []
        if mesh_path is not None:
            candidates.append(mesh_path)
        candidates.extend(self._list_scannet_mesh_candidates())

        last_counts = []
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh.is_empty():
                continue
            vertices = np.asarray(mesh.vertices)
            last_counts.append((path, int(vertices.shape[0])))
            if vertices.shape[0] == seg_indices.shape[0]:
                return mesh, vertices, path

        counts_msg = ", ".join([f"{os.path.basename(p)}:{c}" for p, c in last_counts])
        raise ValueError(
            "segIndices size does not match mesh vertices. "
            f"segIndices={seg_indices.shape[0]}, candidates=({counts_msg})"
        )

    def load_instance_pcd(self, mesh_path=None, agg_path=None, seg_path=None):
        """
        Build an instance-colored point cloud from ScanNet mesh + seg files.

        Returns: open3d.geometry.PointCloud
        """
        if agg_path is None:
            agg_path = self._find_scannet_file([".aggregation.json"])
        if seg_path is None:
            seg_path = self._find_scannet_file([".segs.json"])
        if not agg_path or not os.path.exists(agg_path):
            raise FileNotFoundError("ScanNet aggregation (.aggregation.json) not found")
        if not seg_path or not os.path.exists(seg_path):
            raise FileNotFoundError("ScanNet segs (.segs.json) not found")

        with open(agg_path, "r") as f:
            agg_data = json.load(f)
        with open(seg_path, "r") as f:
            seg_data = json.load(f)

        seg_indices = np.array(seg_data.get("segIndices", []), dtype=np.int64)
        if seg_indices.size == 0:
            raise ValueError("segIndices missing or empty in segs.json")

        mesh, _, mesh_path = self._select_mesh_matching_segs(
            seg_indices, mesh_path=mesh_path
        )

        seg_to_inst = {}
        for obj in agg_data.get("segGroups", []):
            inst_id = int(obj.get("id", -1))
            for seg_id in obj.get("segments", []):
                seg_to_inst[int(seg_id)] = inst_id

        inst_ids = np.array([seg_to_inst.get(int(s), -1) for s in seg_indices])
        unique_ids = np.unique(inst_ids)

        rng = np.random.default_rng(0)
        inst_colors = {int(i): rng.random(3) for i in unique_ids if i >= 0}
        default_color = np.array([0.2, 0.2, 0.2])
        colors = np.array([inst_colors.get(int(i), default_color) for i in inst_ids])

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def get_gt_object_points(self, mesh_path=None, agg_path=None, seg_path=None):
        """Return dict: object_id -> (N,3) points from mesh vertices."""
        if self._gt_object_points is not None:
            return self._gt_object_points

        if agg_path is None:
            agg_path = self._find_scannet_file([".aggregation.json"])
        if seg_path is None:
            seg_path = self._find_scannet_file([".segs.json"])
        if not agg_path or not os.path.exists(agg_path):
            raise FileNotFoundError("ScanNet aggregation (.aggregation.json) not found")
        if not seg_path or not os.path.exists(seg_path):
            raise FileNotFoundError("ScanNet segs (.segs.json) not found")

        with open(agg_path, "r") as f:
            agg_data = json.load(f)
        with open(seg_path, "r") as f:
            seg_data = json.load(f)

        seg_indices = np.array(seg_data.get("segIndices", []), dtype=np.int64)
        if seg_indices.size == 0:
            raise ValueError("segIndices missing or empty in segs.json")

        mesh, vertices, _ = self._select_mesh_matching_segs(
            seg_indices, mesh_path=mesh_path
        )

        seg_to_inst = {}
        for obj in agg_data.get("segGroups", []):
            inst_id = int(obj.get("id", -1))
            for seg_id in obj.get("segments", []):
                seg_to_inst[int(seg_id)] = inst_id

        inst_ids = np.array([seg_to_inst.get(int(s), -1) for s in seg_indices])

        obj_points = {}
        for obj_id in self.gt_objects.keys():
            mask = inst_ids == int(obj_id)
            if np.any(mask):
                obj_points[int(obj_id)] = vertices[mask]
            else:
                obj_points[int(obj_id)] = np.zeros((0, 3), dtype=np.float32)

        self._gt_object_points = obj_points
        return self._gt_object_points

# Example usage:
if __name__ == "__main__":
    cfg = {
        "root_dir": "/home/werby/datasets/ScanNetv2/scans/scene0000_00",
        "depth_scale": 1000.0,
        "depth_min": 0.5,
        "depth_max": 4.0,
    }
    dataset = ScanNetDataset(cfg)
    print(f"Number of samples: {len(dataset)}")

    rgb, depth, pose = dataset[0]
    print(
        f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Pose shape: {pose.shape}"
    )

    scene_pcd = o3d.geometry.PointCloud()
    for i in range(0, len(dataset), 15):
        rgb, depth, pose = dataset[i]
        pcd = dataset.create_pcd(rgb, depth, pose)
        scene_pcd += pcd
    o3d.visualization.draw_geometries([scene_pcd], window_name="ScanNet Scene")
