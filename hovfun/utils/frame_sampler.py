import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List, Optional, Any
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class HDBSCANKeyframeSampler:
    """Helper to select medoid keyframes from HDBSCAN clusters."""

    def __init__(
        self, dataset: Any, selected_indices: Optional[List[int]] = None
    ) -> None:
        self.dataset = dataset
        if selected_indices:
            self.indices = list(selected_indices)
        else:
            self.indices = list(range(len(dataset)))

        poses = []
        for idx in self.indices:
            _, _, pose = self.dataset[idx]
            poses.append(pose)

        self.poses = np.stack(poses, axis=0) if poses else np.empty((0, 4, 4))

    def sample_hdbscan(
        self,
        min_cluster_size=5,
        min_samples=None,
        rot_weight=1.5,
        verbose=True,
    ):
        """
        Hierarchical DBSCAN (HDBSCAN) Sampling.
        Does not require 'eps'. Finds clusters of varying densities.

        Args:
            verbose (bool): Whether to print progress information.
        """
        if verbose:
            print(f"Sampling frames (HDBSCAN): min_cluster_size={min_cluster_size}...")

        translations = self.poses[:, :3, 3]
        quaternions = Rotation.from_matrix(self.poses[:, :3, :3]).as_quat()
        pose_features = np.hstack([translations, rot_weight * quaternions]).astype(
            np.float64
        )
        scaled_features = self._clean_and_scale_features(pose_features, verbose=verbose)

        # HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric="cosine"
        )
        labels = clusterer.fit_predict(scaled_features)

        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        selected_indices = []
        iterator = unique_labels
        if verbose:
            iterator = tqdm(unique_labels, desc="Processing HDBSCAN clusters")
        for label in iterator:
            cluster_indices = np.where(labels == label)[0]

            # Geometric Medoid
            cluster_features = scaled_features[cluster_indices]
            distances = np.linalg.norm(
                cluster_features[:, None, :] - cluster_features[None, :, :], axis=2
            )
            medoid_local_idx = np.argmin(distances.sum(axis=0))
            selected_indices.append(self.indices[cluster_indices[medoid_local_idx]])

        if verbose:
            print(f"HDBSCAN selected {len(selected_indices)} frames.")
        return sorted(selected_indices)

    def _clean_and_scale_features(self, pose_features, verbose=True):
        """
        Clean non-finite values and robustly scale pose features.

        Args:
            pose_features: (N, D) array of pose features.
            verbose (bool): Whether to print warnings.

        Returns:
            Cleaned and scaled feature array.
        """
        # Report non-finite entries
        bad_mask = ~np.isfinite(pose_features)
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            n_rows_bad = int(np.unique(np.where(bad_mask)[0]).size)
            if verbose:
                print(f"[WARN] Non-finite values: elements={n_bad}, rows={n_rows_bad}")

        # Replace +/-inf with NaN for imputation
        pose_features[bad_mask] = np.nan

        # If any column is entirely NaN, fill with zeros
        col_all_nan = np.isnan(pose_features).all(axis=0)
        if np.any(col_all_nan):
            if verbose:
                print(f"[WARN] Columns with all-NaN -> filling with 0.0")
            pose_features[:, col_all_nan] = 0.0

        # Clip extreme outliers per feature to [1, 99] percentile
        q1 = np.nanpercentile(pose_features, 1, axis=0)
        q99 = np.nanpercentile(pose_features, 99, axis=0)
        q1 = np.where(np.isfinite(q1), q1, np.nanmin(pose_features, axis=0))
        q99 = np.where(np.isfinite(q99), q99, np.nanmax(pose_features, axis=0))
        pose_features = np.clip(pose_features, q1, q99)

        # Impute NaNs with median
        imputer = SimpleImputer(strategy="median")
        pose_features = imputer.fit_transform(pose_features)

        # Use robust scaler
        scaler = RobustScaler(quantile_range=(5, 95))
        scaled_features = scaler.fit_transform(pose_features)

        return scaled_features
