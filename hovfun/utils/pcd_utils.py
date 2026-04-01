import open3d as o3d
import numpy as np
from collections import Counter
from scipy.spatial import KDTree
import faiss


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    """
    Denoise the point cloud using DBSCAN.
    :param pcd: Point cloud to denoise.
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: Denoised point cloud.
    """
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def compute_3d_bbox_iou(pcd1, pcd2, padding=0.0):
    """
    Compute 3D Intersection over Union (IoU) between two point clouds.
    In cases where one box is fully contained in the other, this can result in a low IoU.
    To handle this, we also compute the ratio of the intersection volume to the smaller of the two boxes,
    and return the maximum of the two values.
    Args:
        pcd1 (open3d.geometry.PointCloud): Point cloud 1.
        pcd2 (open3d.geometry.PointCloud): Point cloud 2.
        padding (float): Padding to add to the bounding box.
    Returns:
        3D IoU or IoMin between 0 and 1.
    """
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(pcd1.get_min_bound()) - padding
    bbox1_max = np.asarray(pcd1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(pcd2.get_min_bound()) - padding
    bbox2_max = np.asarray(pcd2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    if overlap_volume == 0.0:
        return 0.0

    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    if bbox1_volume == 0.0 or bbox2_volume == 0.0:
        return 0.0

    union_volume = bbox1_volume + bbox2_volume - overlap_volume
    if union_volume == 0.0:
        return 0.0

    iou = overlap_volume / union_volume

    # Intersection over minimum volume (IoMin)
    min_volume = min(bbox1_volume, bbox2_volume)
    if min_volume == 0.0:
        return iou

    iomin = overlap_volume / min_volume

    return max(iou, iomin)


def find_overlapping_ratio_scipy(pcd1, pcd2, radius=0.02):
    """
    Calculate the percentage of overlapping points between two point clouds using SciPy KDTree.

    Parameters:
    pcd1 (open3d.geometry.PointCloud or numpy.ndarray): Point cloud 1.
    pcd2 (open3d.geometry.PointCloud or numpy.ndarray): Point cloud 2.
    radius (float): The radius to consider for a point to be overlapping.

    Returns:
    float: The maximum overlapping ratio between 0 and 1.
    """
    # Convert to numpy arrays if needed
    if isinstance(pcd1, o3d.geometry.PointCloud):
        pcd1_points = np.asarray(pcd1.points)
    else:
        pcd1_points = pcd1

    if isinstance(pcd2, o3d.geometry.PointCloud):
        pcd2_points = np.asarray(pcd2.points)
    else:
        pcd2_points = pcd2

    if pcd1_points.shape[0] == 0 or pcd2_points.shape[0] == 0:
        return 0.0

    # Build KDTree for each point cloud
    tree1 = KDTree(pcd1_points)
    tree2 = KDTree(pcd2_points)

    # Query nearest neighbors within radius
    # For each point in pcd1, find nearest neighbor in pcd2
    distances1, _ = tree2.query(pcd1_points, k=1)
    # For each point in pcd2, find nearest neighbor in pcd1
    distances2, _ = tree1.query(pcd2_points, k=1)

    # Count overlapping points
    overlapping_points1 = np.sum(distances1 < radius)
    overlapping_points2 = np.sum(distances2 < radius)

    # Calculate ratios
    ratio1 = overlapping_points1 / pcd1_points.shape[0]
    ratio2 = overlapping_points2 / pcd2_points.shape[0]

    # Return maximum ratio
    overlapping_ratio = np.max([ratio1, ratio2])

    return overlapping_ratio


def find_overlapping_ratio_faiss(pcd1, pcd2, radius=0.02):
    """
    Calculate the percentage of overlapping points between two point clouds using FAISS.

    Parameters:
    pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
    pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
    radius (float): Radius for KD-Tree query (adjust based on point density).

    Returns:
    float: Overlapping ratio between 0 and 1.
    """
    if type(pcd1) == o3d.geometry.PointCloud and type(pcd2) == o3d.geometry.PointCloud:
        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)

    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return 0

    # Create the FAISS index for each point cloud
    index1 = faiss.IndexFlatL2(pcd1.shape[1])
    index2 = faiss.IndexFlatL2(pcd2.shape[1])
    index1.add(pcd1.astype(np.float32))
    index2.add(pcd2.astype(np.float32))

    # Query all points in pcd1 for nearby points in pcd2
    D1, I1 = index2.search(pcd1.astype(np.float32), k=1)
    D2, I2 = index1.search(pcd2.astype(np.float32), k=1)

    number_of_points_overlapping1 = np.sum(D1 < radius**2)
    number_of_points_overlapping2 = np.sum(D2 < radius**2)

    overlapping_ratio = np.max(
        [
            number_of_points_overlapping1 / pcd1.shape[0],
            number_of_points_overlapping2 / pcd2.shape[0],
        ]
    )

    return overlapping_ratio


def find_overlapping_points_faiss(pcd1, pcd2, radius=0.02):
    """
    Find overlapping points between two point clouds using FAISS.

    Parameters:
    pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
    pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
    radius (float): Radius for considering points as overlapping.

    Returns:
    numpy.ndarray: Indices of overlapping points in pcd1.
    """
    if type(pcd1) == o3d.geometry.PointCloud and type(pcd2) == o3d.geometry.PointCloud:
        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)

    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return np.array([])

    # Create the FAISS index for each point cloud
    index1 = faiss.IndexFlatL2(pcd1.shape[1])
    index2 = faiss.IndexFlatL2(pcd2.shape[1])
    index1.add(pcd1.astype(np.float32))
    index2.add(pcd2.astype(np.float32))

    # Query all points in pcd1 for nearby points in pcd2
    D1, I1 = index2.search(pcd1.astype(np.float32), k=1)
    D2, I2 = index1.search(pcd2.astype(np.float32), k=1)

    overlapping_points_pcd1 = np.where(D1 < radius**2)[0]
    overlapping_points_pcd2 = np.where(D2 < radius**2)[0]

    return overlapping_points_pcd1, overlapping_points_pcd2
