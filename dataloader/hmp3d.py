"""
Habitat Matterport 3D Semantics dataset loader.
"""

import sys
import os
import math
import numpy as np
import cv2
import open3d as o3d
import time

# pylint: disable=all


class HM3DSemDataset:
    """
    Dataset class for the Habitat Matterport3D Semantic dataset.

    This class provides an interface to load RGB-D data samples from the Habitat Matterport3D dataset generated from walks data.
    """

    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            transforms: Optional transformations to apply to the data.
        """
        self.root_dir = cfg.get("root_dir", "")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")
        self.depth_scale = cfg.get("depth_scale", 1000.0)
        self.depth_min = cfg.get("depth_min", 0.0)
        self.depth_max = cfg.get("depth_max", 10.0)
        if self.depth_min < 0 or self.depth_max <= self.depth_min:
            raise ValueError(
                "Invalid depth range: depth_min must be non-negative and less than depth_max."
            )
        self.data_list = self._get_data_list()
        sample_image = self._load_image(self.data_list[0][0])
        self.rgb_H = sample_image.shape[0]  # Height is first dimension
        self.rgb_W = sample_image.shape[1]  # Width is second dimension
        self.depth_intrinsics = self._load_depth_intrinsics(self.rgb_H, self.rgb_W)
        self.name = "HMP3D"
        self.scene_name = self.root_dir.split("/")[-1]

    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image and depth image as numpy arrays.
        """
        rgb_path, depth_path, pose_path = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(pose_path)
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
        rgb_data_list = os.listdir(self.root_dir + "/rgb")
        rgb_data_list = [self.root_dir + "/rgb/" + x for x in rgb_data_list]
        depth_data_list = os.listdir(self.root_dir + "/depth")
        depth_data_list = [self.root_dir + "/depth/" + x for x in depth_data_list]
        pose_data_list = os.listdir(self.root_dir + "/pose")
        pose_data_list = [self.root_dir + "/pose/" + x for x in pose_data_list]
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list.sort()
        if len(rgb_data_list) != len(depth_data_list) or len(rgb_data_list) != len(
            pose_data_list
        ):
            raise ValueError("Mismatch in number of RGB, depth, and pose files.")
        return list(zip(rgb_data_list, depth_data_list, pose_data_list))

    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        # Load the RGB image using OpenCV (BGR format by default)
        rgb_image = cv2.imread(path, cv2.IMREAD_COLOR)
        # Convert BGR to RGB
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
        # Load the depth image using OpenCV (unchanged format to preserve depth values)
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
        with open(path, "r") as file:
            line = file.readline().strip()
            values = line.split()
            values = [float(val) for val in values]
            transformation_matrix = np.array(values).reshape((4, 4))
            C = np.eye(4)
            C[1, 1] = -1
            C[2, 2] = -1
            transformation_matrix = np.matmul(transformation_matrix, C)
        return transformation_matrix

    def _load_depth_intrinsics(self, H, W):
        """
        Load the depth camera intrinsics.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return depth_camera_matrix

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
        # convert rgb and depth images to numpy arrays
        rgb = np.array(rgb)
        depth = np.array(depth)
        # load depth camera intrinsics
        H = rgb.shape[0]
        W = rgb.shape[1]
        camera_matrix = self._load_depth_intrinsics(H, W)
        # create point cloud
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        depth = depth.astype(np.float32) / self.depth_scale
        mask = np.logical_and(depth > self.depth_min, depth < self.depth_max)
        # filter points based on depth mask
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        # Stack pixel coordinates and ones for homogeneous coordinates
        pixels = np.stack([x, y, np.ones_like(x)], axis=0)
        # Invert the camera intrinsics
        K_inv = np.linalg.inv(camera_matrix)
        # Multiply by inverse intrinsics and scale by depth
        xyz = K_inv @ pixels * depth
        X, Y, Z = xyz[0], xyz[1], xyz[2]
        # convert to open3d point cloud
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
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


if __name__ == "__main__":
    # Example usage
    config = {
        "root_dir": "data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi",
        "depth_scale": 1000.0,  # Scale for depth values
        "depth_min": 0.0,  # Minimum depth value
        "depth_max": 3.0,  # Maximum depth value
    }
    dataset = HM3DSemDataset(config)
    print(f"Number of samples: {len(dataset)}")
    K = 1000  # Number of frames to visualize
    scene_pcd_create_pcd = o3d.geometry.PointCloud()
    start_time = time.time()
    for i in range(0, min(K, len(dataset)), 10):
        rgb, depth, pose = dataset[i]
        pcd = dataset.create_pcd(rgb, depth, pose)
        scene_pcd_create_pcd += pcd
    elapsed_time = time.time() - start_time
    print(f"Point cloud creation and accumulation took {elapsed_time:.2f} seconds.")
    o3d.visualization.draw_geometries([scene_pcd_create_pcd])
