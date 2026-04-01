import numpy as np
from PIL import Image
import torch
import math
from typing import Tuple, Optional
from scipy.stats import entropy


def get_image_polar_coords(shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make polar coordinates mask with a fixed size
    """
    h, w = shape
    xs, ys = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    xs = (xs - int(w / 2)) / (w / 2)
    ys = (ys - int(h / 2)) / (h / 2)

    arctans = np.arctan2(ys, xs)
    module = np.sqrt(np.power(xs, 2) + np.power(ys, 2))

    return torch.tensor(module, dtype=torch.float32), torch.tensor(
        arctans, dtype=torch.float32
    )


M_1920_1440, A_1920_1440 = get_image_polar_coords((1920, 1440))
M_1440_1920, A_1440_1920 = get_image_polar_coords((1440, 1920))
M_1080_1920, A_1080_1920 = get_image_polar_coords((1080, 1920))
M_720_1280, A_720_1280 = get_image_polar_coords((720, 1280))
M_680_1200, A_680_1200 = get_image_polar_coords((680, 1200))
M_968_1296, A_968_1296 = get_image_polar_coords((968, 1296))


def score_mask_by_distance_and_size(mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculates normalized scores for a mask's size and its distance from the boundary.

    Args:
        mask (np.ndarray): A 2D NumPy array representing the binary mask.

    Returns:
        A tuple containing:
        - boundary_score (float): A score from 0.0 to 1.0, proportional to the
          minimum distance of the mask from any image edge. 0 means it touches
          the boundary, 1 means its closest point is the image center.
        - size_score (float): A score from 0.0 to 1.0, representing the
          fraction of the image covered by the mask.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")

    h, w = mask.shape
    if h == 0 or w == 0:
        return 0.0, 0.0

    # Find coordinates of all 'on' pixels
    on_pixels = np.argwhere(mask == 1)

    # If the mask is empty, both scores are 0
    if on_pixels.size == 0:
        return 0.0, 0.0

    # --- 1. Boundary Distance Score ---
    # Calculate the distance for each 'on' pixel to all four edges
    y_coords = on_pixels[:, 0]
    x_coords = on_pixels[:, 1]
    dist_to_top = y_coords
    dist_to_bottom = h - 1 - y_coords
    dist_to_left = x_coords
    dist_to_right = w - 1 - x_coords

    # The distance for a pixel is the minimum of its distances to the four edges
    min_pixel_distances = np.minimum.reduce(
        [dist_to_top, dist_to_bottom, dist_to_left, dist_to_right]
    )

    # The mask's distance is the smallest distance found among all its pixels
    min_mask_distance = np.min(min_pixel_distances)

    # Normalize the score. The max possible distance is to the center of the image.
    max_possible_dist = min(h // 2, w // 2)
    boundary_score = (
        min_mask_distance / max_possible_dist if max_possible_dist > 0 else 0.0
    )

    # --- 2. Size Score ---
    mask_size = on_pixels.shape[0]
    total_pixels = mask.size
    size_score = mask_size / total_pixels

    return float(boundary_score), float(size_score)


def get_mask_score(
    mask: np.ndarray, n_bins: Optional[int] = 30
) -> Tuple[float, float, float, float]:
    """
    Computes polar coordinate, boundary, and size scores for the mask.

    Returns:
        A tuple of (arc_score, mod_score, boundary_score, size_score).
        All scores are normalized or designed to be on a comparable scale.
    """
    # Select pre-computed polar coordinates based on mask shape
    if mask.shape == (1920, 1440):
        module, arctans = M_1920_1440, A_1920_1440
    elif mask.shape == (1440, 1920):
        module, arctans = M_1440_1920, A_1440_1920
    elif mask.shape == (1080, 1920):
        module, arctans = M_1080_1920, A_1080_1920
    elif mask.shape == (720, 1280):
        module, arctans = M_720_1280, A_720_1280
    elif mask.shape == (680, 1200):
        module, arctans = M_680_1200, A_680_1200
    elif mask.shape == (968, 1296):
        module, arctans = M_968_1296, A_968_1296
    else:
        module, arctans = get_image_polar_coords(mask.shape)

    # Convert PyTorch tensor to NumPy array if needed
    mask_np = mask
    if hasattr(mask, "detach"):  # Check if it's a PyTorch tensor
        mask_np = mask.detach().cpu().numpy()

    # --- Boundary and Size Scores ---
    boundary_score, size_score = score_mask_by_distance_and_size(mask_np)

    # Convert mask to torch tensor for polar coordinate calculations
    mask_tensor = torch.tensor(mask_np) if not torch.is_tensor(mask) else mask

    # --- Polar Coordinate Scores ---
    if torch.count_nonzero(mask_tensor) > 0:
        m_arctans = arctans[mask_tensor == 1]
        m_mod = module[mask_tensor == 1]

        hist_arc, _ = torch.histogram(
            m_arctans, bins=n_bins, range=(-torch.pi, torch.pi)
        )
        hist_mod, bins_mod = torch.histogram(
            m_mod, bins=n_bins, range=(0, math.sqrt(2))
        )

        # Normalize histograms to be probability distributions
        hist_arc_norm = hist_arc.float() / torch.sum(hist_arc).float()
        hist_mod_norm = hist_mod.float() / torch.sum(hist_mod).float()

        # Ideal distributions
        arc_dist = torch.full((n_bins,), 1.0 / n_bins)  # Uniform distribution for angle

        mod_dist = torch.zeros(n_bins)
        max_mod = torch.max(m_mod)
        # Find the bin that contains the maximum radius
        max_bin_idx = torch.searchsorted(bins_mod, max_mod, right=True) - 1
        max_bin_idx = torch.clamp(max_bin_idx, 0, n_bins - 1)
        mod_dist[0 : max_bin_idx + 1] = 1.0
        mod_dist = mod_dist / torch.sum(mod_dist)  # Normalize ideal radius distribution

        # Calculate KL divergence (entropy)
        arc_score = entropy(hist_arc_norm.numpy(), arc_dist.numpy())
        mod_score = entropy(hist_mod_norm.numpy(), mod_dist.numpy())
    else:
        # If mask is empty, all scores are 0
        arc_score, mod_score = 0.0, 0.0

    return arc_score, mod_score, boundary_score, size_score * 25


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (
            ~mask_sub[contained_idx[1][i]]
        )

    return mask_sub


def crop_image(img, bbox):
    """
    Crop an image (numpy ndarray or PIL Image) using a bounding box.

    Args:
        img: np.ndarray or PIL.Image.Image
        bbox: list or array-like of [x_min, y_min, x_max, y_max]

    Returns:
        Cropped image (same type as input)
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    if isinstance(img, np.ndarray):
        return img[y_min:y_max, x_min:x_max]
    elif isinstance(img, Image.Image):
        return img.crop((x_min, y_min, x_max, y_max))
    else:
        raise TypeError("img must be a numpy ndarray or PIL Image")
