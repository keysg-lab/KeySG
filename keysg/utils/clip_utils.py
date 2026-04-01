from typing import List, Dict, Any, Optional

import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image


# Default configuration
DEFAULT_CLIP_CONFIG = {
    # "model_name": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
    # "model_name": "ViT-B-32",
    # "model_name": "ViT-H-14",
    "model_name": "hf-hub:timm/ViT-gopt-16-SigLIP2-384",
    "pretrained": "laion2b_s32b_b79k",
    "device": "cuda",
}


class CLIPFeatureExtractor:
    """
    CLIP Feature Extractor class for extracting image and text features.

    Example:
        config = {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "device": "cuda",
        }
        clip_extractor = CLIPFeatureExtractor(config)

        # Extract image features
        image = np.array(Image.open("image.jpg"))
        img_features = clip_extractor.get_img_feats(image)

        # Extract text features
        text_features = clip_extractor.get_text_feats(["a cat", "a dog"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CLIP model and preprocessing.

        Args:
            config: Configuration dictionary with model parameters
        """
        # Use default config if none provided
        if config is None:
            config = DEFAULT_CLIP_CONFIG.copy()
        else:
            # Merge with defaults
            full_config = DEFAULT_CLIP_CONFIG.copy()
            full_config.update(config)
            config = full_config

        self.model_name = config.get("model_name", None)
        self.pretrained = config.get("pretrained", None)

        # Set device
        if config["device"] is not None:
            self.device = config["device"]
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available() and not config["force_cpu"]
                else "cpu"
            )

        # Initialize CLIP model
        self._load_model()

    def _load_model(self):
        """Load CLIP model, preprocess function, and tokenizer."""
        print(f"Loading CLIP model: {self.model_name}")

        # For hf-hub models, don't pass pretrained parameter
        if "hf-hub:" in self.model_name:
            self.model, self.preprocess = create_model_from_pretrained(self.model_name)
            self.tokenizer = get_tokenizer(self.model_name)
        else:
            self.model, self.preprocess = create_model_from_pretrained(
                self.model_name, pretrained=self.pretrained
            )
            # Get tokenizer based on model name
            if "ViT-H-14" in self.model_name:
                self.tokenizer = get_tokenizer("ViT-H-14")
            elif "ViT-L-14" in self.model_name:
                self.tokenizer = get_tokenizer("ViT-L-14")
            elif "ViT-B-32" in self.model_name:
                self.tokenizer = get_tokenizer("ViT-B-32")
            elif "ViT-B-16" in self.model_name:
                self.tokenizer = get_tokenizer("ViT-B-16")
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)

        # Get feature dimension
        if hasattr(self.model.visual, "output_dim"):
            self.clip_feat_dim = self.model.visual.output_dim
        elif hasattr(self.model.text, "output_dim"):
            self.clip_feat_dim = self.model.text.output_dim
        else:
            raise ValueError("Could not determine CLIP feature dimension.")
        print(f"CLIP model loaded successfully on {self.device}")
        print(f"Feature dimension: {self.clip_feat_dim}")

    def get_img_feats(self, img: np.ndarray | Image.Image) -> np.ndarray:
        """
        Get image features from CLIP model.

        Args:
            img: Image as numpy array (H, W, C)

        Returns:
            Image features as numpy array (clip_feat_dim,)
        """
        img_pil = Image.fromarray(np.uint8(img)) if isinstance(img, np.ndarray) else img
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            img_feats = self.model.encode_image(img_tensor)
            img_feats = F.normalize(img_feats, dim=-1)

        return img_feats.cpu().numpy()[0]

    def get_img_feats_batch(
        self, imgs: List[np.ndarray | Image.Image], batch_size: int = 64
    ) -> np.ndarray:
        """
        Get image features for a batch of images.

        Args:
            imgs: List of images as numpy arrays
            batch_size: Batch size for processing

        Returns:
            Image features as numpy array (N, clip_feat_dim)
        """
        imgs_feats = np.zeros((len(imgs), self.clip_feat_dim))

        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for img in batch_imgs:
                img_pil = (
                    Image.fromarray(np.uint8(img))
                    if isinstance(img, np.ndarray)
                    else img
                )
                if img_pil is None:
                    continue
                img_tensor = self.preprocess(img_pil)
                batch_tensors.append(img_tensor)

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                batch_feats = self.model.encode_image(batch_tensor)
                batch_feats = F.normalize(batch_feats, dim=-1)

            imgs_feats[i : i + len(batch_imgs)] = batch_feats.cpu().numpy()

        return imgs_feats

    def get_text_feats(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Get text features from CLIP model.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Text features as numpy array (N, clip_feat_dim)
        """
        text_tokens = self.tokenizer(
            texts, context_length=self.model.context_length
        ).to(self.device)
        text_feats = np.zeros((len(texts), self.clip_feat_dim), dtype=np.float32)

        for i in range(0, len(text_tokens), batch_size):
            batch_tokens = text_tokens[i : i + batch_size]

            with torch.no_grad(), torch.amp.autocast("cuda"):
                batch_feats = self.model.encode_text(batch_tokens)
                batch_feats = F.normalize(batch_feats, dim=-1)

            text_feats[i : i + len(batch_tokens)] = batch_feats.cpu().numpy()

        return text_feats

    def match_text_to_imgs(
        self, text: str, images: List[np.ndarray | Image.Image]
    ) -> tuple:
        """
        Match text to images and return similarity scores.

        Args:
            text: Text query
            images: List of images as numpy arrays or PIL Images

        Returns:
            Tuple of (scores, img_features, text_features)
        """
        img_feats = self.get_img_feats_batch(images)
        text_feats = self.get_text_feats([text])

        scores = img_feats @ text_feats.T
        scores = scores.squeeze()

        return scores, img_feats, text_feats

    def get_nearest_images(
        self,
        text_feats: np.ndarray,
        img_feats: np.ndarray,
        images: List[np.ndarray | Image.Image],
    ) -> tuple:
        """
        Get images sorted by similarity to text features.

        Args:
            text_feats: Text features (1, clip_feat_dim)
            img_feats: Image features (N, clip_feat_dim)
            images: List of images

        Returns:
            Tuple of (sorted_indices, sorted_images, sorted_scores)
        """
        scores = img_feats @ text_feats.T
        scores = scores.squeeze()

        # Sort from high to low similarity
        sorted_indices = np.argsort(scores)[::-1]
        sorted_images = [images[i] for i in sorted_indices]
        sorted_scores = np.sort(scores)[::-1]

        return sorted_indices, sorted_images, sorted_scores

    def compute_text_probabilities(
        self, image_features: np.ndarray, text_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute text probabilities for given image and text features using logit scale and bias.

        Args:
            image_features: Image features (1, clip_feat_dim)
            text_features: Text features (N, clip_feat_dim)

        Returns:
            Text probabilities as numpy array (N,)
        """
        # Convert to torch tensors
        img_feats = torch.from_numpy(image_features).to(self.device)
        txt_feats = torch.from_numpy(text_features).to(self.device)

        with torch.no_grad():
            # Compute similarity scores with logit scale and bias
            similarities = (
                img_feats @ txt_feats.T * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
            probabilities = torch.sigmoid(similarities)

        return probabilities.cpu().numpy().squeeze()

    def match_text_to_imgs_with_probs(
        self, texts: List[str], images: List[np.ndarray]
    ) -> tuple:
        """
        Match texts to images and return probabilities instead of raw similarities.

        Args:
            texts: List of text queries
            images: List of images as numpy arrays

        Returns:
            Tuple of (probabilities, img_features, text_features)
        """
        img_feats = self.get_img_feats_batch(images)
        text_feats = self.get_text_feats(texts)

        # Convert to torch tensors
        img_feats_torch = torch.from_numpy(img_feats).to(self.device)
        txt_feats_torch = torch.from_numpy(text_feats).to(self.device)

        with torch.no_grad():
            # Compute similarity scores with logit scale and bias
            similarities = (
                img_feats_torch @ txt_feats_torch.T * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
            probabilities = torch.sigmoid(similarities)

        return probabilities.cpu().numpy(), img_feats, text_feats

    def feats_denoise_dbscan(self, feats, eps=0.02, min_points=2):
        """
        Denoise the features using DBSCAN
        :param feats: Features to denoise.
        :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: Denoised features.
        """
        # Convert to numpy arrays
        feats = np.array(feats)
        # Create DBSCAN object
        clustering = DBSCAN(eps=eps, min_samples=min_points, metric="cosine").fit(feats)

        # Get the labels
        labels = clustering.labels_

        # Count all labels in the cluster
        counter = Counter(labels)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            # Create mask for points in the largest cluster
            largest_mask = labels == most_common_label
            # Apply mask
            largest_cluster_feats = feats[largest_mask]
            feats = largest_cluster_feats
            # take the feature with the highest similarity to the mean of the cluster
            if len(feats) > 1:
                mean_feats = np.mean(largest_cluster_feats, axis=0)
                # similarity = np.dot(largest_cluster_feats, mean_feats)
                # max_idx = np.argmax(similarity)
                # feats = feats[max_idx]
                feats = mean_feats
        else:
            feats = np.mean(feats, axis=0)
        return feats


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "model_name": "hf-hub:timm/ViT-gopt-16-SigLIP2-384",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Initialize CLIP extractor
    clip_extractor = CLIPFeatureExtractor(config)

    # Example usage (you would need actual images)
    print("CLIP Feature Extractor initialized successfully!")
    print(f"Model: {clip_extractor.model_name}")
    print(f"Feature dimension: {clip_extractor.clip_feat_dim}")
    print(f"Device: {clip_extractor.device}")

    # Example text feature extraction
    texts = ["a cat", "a dog", "a car"]
    text_features = clip_extractor.get_text_feats(texts)
    print(f"Text features shape: {text_features.shape}")
