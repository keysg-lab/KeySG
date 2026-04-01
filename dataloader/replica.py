import sys
import os
import math
import numpy as np
import cv2
import open3d as o3d
import json


class ReplicaDataset:
    """
    Dataset class for the Replica dataset, matching the interface of HM3DSemDataset.
    """

    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            transforms: Optional transformations to apply to the data.
        """
        self.root_dir = cfg.get("root_dir", "")
        self.depth_scale = cfg.get("depth_scale", 1000.0)
        self.depth_min = cfg.get("depth_min", 0.5)
        self.depth_max = cfg.get("depth_max", 4.0)
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")
        self.transforms = cfg.get("transforms", None)
        # Load intrinsics and scale
        cam_params_path = os.path.join(
            os.path.split(self.root_dir)[0], "cam_params.json"
        )
        self.depth_intrinsics, self.depth_scale = self._load_depth_intrinsics(
            cam_params_path
        )
        self.data_list = self._get_data_list()
        # Get image shape
        sample_image = self._load_image(self.data_list[0][0])
        self.rgb_H = sample_image.shape[0]
        self.rgb_W = sample_image.shape[1]
        self.name = "Replica"
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
        pose = self._load_pose(pose_path, idx)
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
        return rgb_image, depth_image, pose

    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path, pose path).
        """
        rgb_dir = os.path.join(self.root_dir, "results")
        rgb_data_list = []
        depth_data_list = []
        pose_path = os.path.join(self.root_dir, "traj.txt")
        for file in os.listdir(rgb_dir):
            if file.startswith("frame"):
                rgb_data_list.append(os.path.join(rgb_dir, file))
            elif file.startswith("depth"):
                depth_data_list.append(os.path.join(rgb_dir, file))
        rgb_data_list.sort()
        depth_data_list.sort()
        # pose_path is the same for all frames, but we need to keep the index
        pose_data_list = [pose_path] * len(rgb_data_list)
        if len(rgb_data_list) != len(depth_data_list):
            raise ValueError("Mismatch in number of RGB and depth files.")
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

    def _load_pose(self, path, idx):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        with open(path, "r") as file:
            lines = file.readlines()
            if 0 <= idx < len(lines):
                line = lines[idx]
                values = [float(val) for val in line.split()]
                # Reshape the 16 values into a 4x4 matrix
                transformation_matrix = np.array(values).reshape((4, 4))
                return transformation_matrix

    def _load_depth_intrinsics(self, path):
        """
        Load the depth camera intrinsics from the given path.

        Args:
            path: Path to the depth camera intrinsics file.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix) and scale.
        """
        with open(path, "r") as file:
            data = json.load(file)
            camera_params = data.get("camera")
            if camera_params:
                fx = camera_params.get("fx")
                fy = camera_params.get("fy")
                cx = camera_params.get("cx")
                cy = camera_params.get("cy")
                scale = camera_params.get("scale")
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                return K, scale
        raise ValueError("Camera parameters not found in cam_params.json")

    def create_pcd(self, rgb, depth, camera_pose=None):
        """
        Create a point cloud from RGB-D images.

        Args:
            rgb: RGB image as a numpy array.
            depth: Depth image as a numpy array.
            camera_pose: Camera pose as a numpy array (4x4 matrix).

        Returns:
            Point cloud as an Open3D object.
        """
        rgb = np.array(rgb)
        depth = np.array(depth)
        H, W = rgb.shape[0], rgb.shape[1]
        camera_matrix = self.depth_intrinsics
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        depth = depth.astype(np.float32) / self.depth_scale
        mask = np.logical_and(depth > self.depth_min, depth < self.depth_max)
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        pixels = np.stack([x, y, np.ones_like(x)], axis=0)
        K_inv = np.linalg.inv(camera_matrix)
        xyz = K_inv @ pixels * depth
        X, Y, Z = xyz[0], xyz[1], xyz[2]
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
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
        H, W = mask_2d.shape
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Apply mask
        mask_indices = mask_2d > 0
        x_masked = x[mask_indices]
        y_masked = y[mask_indices]

        # Get depth values
        depth = depth_image.astype(np.float32) / self.depth_scale
        depth_masked = depth[mask_indices]

        # Get RGB values for masked pixels
        rgb_masked = rgb_image[mask_indices]

        # Filter by depth range
        valid_depth = np.logical_and(
            depth_masked > self.depth_min, depth_masked < self.depth_max
        )
        x_valid = x_masked[valid_depth]
        y_valid = y_masked[valid_depth]
        depth_valid = depth_masked[valid_depth]
        rgb_valid = rgb_masked[valid_depth]

        if len(x_valid) == 0:
            # Return empty point cloud
            return o3d.geometry.PointCloud()

        # Convert to 3D coordinates using dataset's intrinsics
        pixels = np.stack([x_valid, y_valid, np.ones_like(x_valid)], axis=0)
        K_inv = np.linalg.inv(self.depth_intrinsics)
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


# Example usage:
if __name__ == "__main__":
    cfg = {
        "root_dir": "/home/werby/datasets/Replica_RGBD/Replica/room0",
        "depth_scale": 1000.0,
        "depth_min": 0.5,
        "depth_max": 4.0,
    }
    dataset = ReplicaDataset(cfg)
    print(f"Number of samples: {len(dataset)}")

    scene_pcd = o3d.geometry.PointCloud()
    for i in range(0, 500, 10):
        rgb, depth, pose = dataset[i]
        pcd = dataset.create_pcd(rgb, depth, pose)
        scene_pcd += pcd
    o3d.visualization.draw_geometries([scene_pcd], window_name="Replica Scene")
