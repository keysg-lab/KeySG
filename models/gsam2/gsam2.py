import sys
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from supervision.draw.color import ColorPalette
from PIL import Image
import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
from ram.models import ram_plus, ram, tag2text
from ram import inference_ram as inference
from ram import get_transform
from ultralytics import YOLOWorld, RTDETR

# Make project root importable before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local scene-understanding VLM tagger (optional alternative to RAM++)
from models.llm.gpt_vlm import GPT_VLMInterface as VLMInterface  # unified VLM client
from hovfun.utils.img_utils import mask_subtract_contained

COLOR_PALETTES = {
    "default": None,  # Use supervision default
    "bright": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"],
    "pastel": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA", "#E0BBE4"],
    "dark": ["#8B0000", "#006400", "#000080", "#8B8000", "#8B008B", "#008B8B"],
    "high_contrast": ["#FF0000", "#FFFFFF", "#000000", "#FFFF00", "#00FF00", "#0000FF"],
}


def get_color_palette(palette_name="default"):
    """Get a predefined color palette."""
    if palette_name not in COLOR_PALETTES:
        raise ValueError(
            f"Unknown palette: {palette_name}. Available: {list(COLOR_PALETTES.keys())}"
        )

    return COLOR_PALETTES[palette_name]


class GroundingSAM2:
    """
    A wrapper class for SAM2 + a single selected detection backend.

    This class provides a clean API for object detection and segmentation using
    SAM2 for precise segmentation and exactly one detection backend per instance
    (Grounding DINO, OwlV2, YOLO-World, functional elements, or pure SAM2 grid).

    Example:
        # Grounding DINO instance (loads only GDINO + SAM2)
        gsam = GroundingSAM2(
            detection_mode="grounding_dino",
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            grounding_model_id="IDEA-Research/grounding-dino-base",
        )
        results = gsam.predict(image="path/to/img.jpg", text_prompt="car. tire.")
        gsam.visualize_results(results, image="path/to/img.jpg", output_path="out.jpg")

        # To switch backend, create a new instance (saves GPU memory)
        gsam_yolo = GroundingSAM2(
            detection_mode="yolo_world",
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            yolo_world_checkpoint="./checkpoints/yolov8x-worldv2.pt",
        )

        Tagging backends:
        - tags_backend="ram" (default) uses RAM++ checkpoint for tags
        - tags_backend="vlm" uses the VLM-based tagger from hovfun.graph.scene_understanding
    """

    def __init__(
        self,
        detection_mode: str,
        sam2_checkpoint: str,
        sam2_model_config: str,
        # Detection backend specific config: Only pass what your mode needs
        grounding_model_id: Optional[str] = None,
        owlv2_model_id: Optional[str] = None,
        llmdet_model_id: Optional[str] = None,
        yolo_world_checkpoint: Optional[str] = None,
        fungraph_checkpoint: Optional[str] = None,
        ram_pretrained: Optional[str] = None,
        ram_image_size: int = 384,
        device: Optional[str] = None,
        force_cpu: bool = False,
        llmdet_max_tags_per_batch: int = 30,
    ):
        """
        Initialize GroundingSAM2 with a single selected detection backend.

        Args:
            detection_mode: One of ['grounding_dino', 'owlv2', 'llmdet', 'yolo_world', 'sam2_grid', 'functional_elements']
            sam2_checkpoint: Path to SAM2 checkpoint file
            sam2_model_config: Path to SAM2 model configuration
            grounding_model_id: HF model id for Grounding DINO (when detection_mode='grounding_dino')
            owlv2_model_id: HF model id for OwlV2 (when detection_mode='owlv2')
            llmdet_model_id: HF model id for LLMDet (when detection_mode='llmdet')
            yolo_world_checkpoint: Path to YOLO-World checkpoint (when detection_mode='yolo_world')
            fungraph_checkpoint: Path to functional element detector checkpoint (when detection_mode='functional_elements')
            ram_pretrained: Optional RAM++ checkpoint for tag_image() (lazy loaded)
            ram_image_size: Image size used by RAM++ when loaded
            device: 'cuda' or 'cpu' (auto if None)
            force_cpu: Force CPU even if CUDA present
            llmdet_max_tags_per_batch: Maximum number of tags to process per batch for LLMDet mode
        """
        self.detection_mode = detection_mode

        # Store minimal config
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config

        self.grounding_model_id = grounding_model_id
        self.owlv2_model_id = owlv2_model_id
        self.llmdet_model_id = llmdet_model_id
        self.yolo_world_checkpoint = yolo_world_checkpoint
        self.fungraph_checkpoint = fungraph_checkpoint

        # LLMDet batching config
        self.llmdet_max_tags_per_batch = llmdet_max_tags_per_batch

        # RAM/VLM tagging config (lazy)
        self.ram_pretrained = ram_pretrained
        self.ram_image_size = ram_image_size
        self._ram_loaded = False
        self._vlm_client = None

        # Set device
        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
            )

        # Light environment tweaks
        self._setup_environment()

        # Always load SAM2, and only the selected detection model
        self._load_sam2()
        self._load_selected_detection_model()

    def _setup_environment(self):
        """Setup PyTorch environment for optimal performance (non-intrusive)."""
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _load_sam2(self):
        """Load SAM2 predictor and mask generator (always needed)."""
        sam2_model = build_sam2(
            self.sam2_model_config, self.sam2_checkpoint, device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=10,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )

    def _load_selected_detection_model(self):
        """Load only the detection backend requested by detection_mode."""
        mode = self.detection_mode
        if mode == "grounding_dino":
            if not self.grounding_model_id:
                raise ValueError(
                    "grounding_model_id must be provided for grounding_dino mode"
                )
            self.processor = AutoProcessor.from_pretrained(self.grounding_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_model_id
            ).to(self.device)
        elif mode == "owlv2":
            if not self.owlv2_model_id:
                raise ValueError("owlv2_model_id must be provided for owlv2 mode")
            self.owlv2_processor = Owlv2Processor.from_pretrained(self.owlv2_model_id)
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                self.owlv2_model_id
            ).to(self.device)
        elif mode == "llmdet":
            if not self.llmdet_model_id:
                raise ValueError("llmdet_model_id must be provided for llmdet mode")
            self.llmdet_processor = AutoProcessor.from_pretrained(self.llmdet_model_id)
            self.llmdet_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.llmdet_model_id
            ).to(self.device)
        elif mode == "yolo_world":
            if not self.yolo_world_checkpoint:
                raise ValueError(
                    "yolo_world_checkpoint must be provided for yolo_world mode"
                )
            self.yolo_world_model = YOLOWorld(self.yolo_world_checkpoint)
            self.yolo_world_model.eval()
        elif mode == "functional_elements":
            if not self.fungraph_checkpoint:
                raise ValueError(
                    "fungraph_checkpoint must be provided for functional_elements mode"
                )
            self.fungraph_model = RTDETR(self.fungraph_checkpoint)
            self.fungraph_model.eval()
        elif mode == "sam2_grid":
            # No extra detection model required
            pass
        else:
            raise ValueError(
                "Unknown detection_mode. Choose from ['grounding_dino','owlv2','llmdet','yolo_world','sam2_grid','functional_elements']"
            )

    def _ensure_ram_loaded(self):
        """Lazy-load RAM++ for tag_image only when requested."""
        if self._ram_loaded:
            return
        if not self.ram_pretrained:
            raise ValueError("ram_pretrained must be provided to use tag_image()")
        self.ram_transform = get_transform(image_size=self.ram_image_size)
        self.ram_model = ram_plus(
            pretrained=self.ram_pretrained,
            image_size=self.ram_image_size,
            vit="swin_l",
        )
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.device)
        self._ram_loaded = True

    def _ensure_vlm_client(self):
        """Lazy-create the VLMInterface client for VLM-based tagging."""
        if self._vlm_client is None:
            # Default to local Ollama for both vision and text; can be extended later
            self._vlm_client = VLMInterface()

    def _generate_grid_points(
        self, image_size: Tuple[int, int], grid_size: int = 32
    ) -> np.ndarray:
        """
        Generate a grid of points for segmenting everything in the image.

        Args:
            image_size: (width, height) of the image
            grid_size: Number of points per side of the grid

        Returns:
            Grid points as numpy array of shape (N, 2)
        """
        width, height = image_size
        x_points = np.linspace(0, width - 1, grid_size)
        y_points = np.linspace(0, height - 1, grid_size)
        xx, yy = np.meshgrid(x_points, y_points)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        return grid_points

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        mode: Optional[str] = None,
        text_prompt: Optional[str] = None,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        multimask_output: bool = False,
        grid_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Perform object detection and segmentation on an image using different modes.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            mode: Optional override. If provided and different from the instance's mode, an error is raised.
            text_prompt: Text description for grounding_dino or owlv2 mode (should end with dots)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching (grounding_dino only)
            multimask_output: Whether to return multiple masks per object
            grid_size: Grid size for sam2_grid mode

        Returns:
            Dictionary containing detection results with keys:
            - 'boxes': Bounding boxes (N, 4) in xyxy format
            - 'masks': Segmentation masks (N, H, W)
            - 'scores': Confidence scores (N,)
            - 'labels': Class labels (N,)
            - 'class_ids': Class IDs (N,)
            - 'image_size': (width, height)
        """
        # Load and process image
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        # Set image for SAM2
        self.sam2_predictor.set_image(np.array(pil_image.convert("RGB")))
        effective_mode = mode if mode is not None else self.detection_mode
        if mode is not None and mode != self.detection_mode:
            raise ValueError(
                "This instance was created for detection_mode='{}' but got mode='{}'. Create a new GroundingSAM2 for another backend.".format(
                    self.detection_mode, mode
                )
            )

        if effective_mode == "grounding_dino":
            return self._predict_grounding_dino(
                pil_image, text_prompt, box_threshold, text_threshold, multimask_output
            )
        elif effective_mode == "owlv2":
            return self._predict_owlv2(
                pil_image, text_prompt, box_threshold, multimask_output
            )
        elif effective_mode == "yolo_world":
            return self._predict_yolo_world(pil_image, box_threshold, multimask_output)
        elif effective_mode == "sam2_grid":
            return self._predict_sam2_grid(pil_image, grid_size, multimask_output)
        elif effective_mode == "functional_elements":
            return self._predict_functional_elements(
                pil_image, box_threshold, multimask_output
            )
        elif effective_mode == "llmdet":
            return self._predict_llmdet(
                pil_image, text_prompt, box_threshold, multimask_output
            )
        else:
            raise ValueError(
                f"Unknown mode: {effective_mode}. Available modes: 'grounding_dino', 'owlv2', 'llmdet', 'yolo_world', 'sam2_grid', 'functional_elements'"
            )

    def _predict_grounding_dino(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Predict using Grounding DINO + SAM2."""
        if text_prompt is None:
            raise ValueError("text_prompt is required for grounding_dino mode")

        # Ensure text prompt format (lowercase + ends with dot)
        if not text_prompt.endswith("."):
            text_prompt += "."
        text_prompt = text_prompt.lower()
        # Run Grounding DINO
        inputs = self.processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        grounding_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )

        if not grounding_results or len(grounding_results[0]["boxes"]) == 0:
            # Return empty results if no detections
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        # Get boxes for SAM2
        input_boxes = grounding_results[0]["boxes"].cpu().numpy()

        # Run SAM2 segmentation
        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Subtract masks contained within other masks
        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        # Extract results
        confidences = grounding_results[0]["scores"].cpu().numpy()
        class_names = grounding_results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": class_names,
            "class_ids": class_ids,
            "image_size": pil_image.size,
        }

    def _predict_owlv2(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Predict using OwlV2 + SAM2."""
        if text_prompt is None:
            raise ValueError("text_prompt is required for owlv2 mode")

        # OwlV2 expects a list of lists of texts
        texts = [[p.strip() for p in text_prompt.split(".") if p.strip()]]

        # Run OwlV2
        inputs = self.owlv2_processor(
            text=texts, images=pil_image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        owl_results = self.owlv2_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=box_threshold
        )

        # Retrieve predictions for the first image
        i = 0
        if not owl_results or len(owl_results[i]["boxes"]) == 0:
            # Return empty results if no detections
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        input_boxes = owl_results[i]["boxes"].cpu().numpy()
        confidences = owl_results[i]["scores"].cpu().numpy()
        owl_labels = owl_results[i]["labels"].cpu().numpy()

        # Get class names from the original text prompt
        class_names = [texts[i][label] for label in owl_labels]
        class_ids = owl_labels

        # Run SAM2 segmentation
        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": class_names,
            "class_ids": class_ids,
            "image_size": pil_image.size,
        }

    def _predict_llmdet(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Predict using LLMDet + SAM2 with batching for large tag lists.

        LLMDet has a limit on the number of tags it can process at once.
        This function automatically batches the tags and merges results,
        resolving overlapping masks by keeping the highest confidence ones.
        """
        if text_prompt is None:
            raise ValueError("text_prompt is required for llmdet mode")

        # Prepare texts as list of strings split on '.'
        tags = [p.strip() for p in text_prompt.split(".") if p.strip()]

        if len(tags) <= self.llmdet_max_tags_per_batch:
            # Process single batch
            return self._predict_llmdet_batch(
                pil_image, [tags], box_threshold, multimask_output
            )

        # Process multiple batches
        all_boxes, all_masks, all_scores, all_labels, all_class_ids = [], [], [], [], []

        for i in range(0, len(tags), self.llmdet_max_tags_per_batch):
            batch_tags = tags[i : i + self.llmdet_max_tags_per_batch]
            batch_results = self._predict_llmdet_batch(
                pil_image, [batch_tags], box_threshold, multimask_output
            )

            if len(batch_results["boxes"]) > 0:
                all_boxes.append(batch_results["boxes"])
                all_masks.append(batch_results["masks"])
                all_scores.append(batch_results["scores"])
                all_labels.extend(batch_results["labels"])
                # Adjust class_ids to be unique across batches
                base_id = len(all_class_ids)
                adjusted_class_ids = batch_results["class_ids"] + base_id
                all_class_ids.extend(adjusted_class_ids)

        if not all_boxes:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        # Merge results and resolve overlaps
        merged_boxes = np.vstack(all_boxes)
        merged_masks = np.vstack(all_masks)
        merged_scores = np.hstack(all_scores)
        merged_class_ids = np.array(all_class_ids)

        # Resolve overlapping masks by keeping highest confidence
        final_indices = self._resolve_mask_overlaps(merged_masks, merged_scores)

        return {
            "boxes": merged_boxes[final_indices],
            "masks": merged_masks[final_indices].astype(bool),
            "scores": merged_scores[final_indices],
            "labels": [all_labels[i] for i in final_indices],
            "class_ids": merged_class_ids[final_indices],
            "image_size": pil_image.size,
        }

    def _predict_llmdet_batch(
        self,
        pil_image: Image.Image,
        texts: List[List[str]],
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Process a single batch of tags with LLMDet."""
        # Run LLMDet
        inputs = self.llmdet_processor(
            images=pil_image, text=texts, return_tensors="pt",
            truncation=True, max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llmdet_model(**inputs)

        # Target sizes (H, W)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)

        # Post-process
        llm_results = self.llmdet_processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=box_threshold, target_sizes=target_sizes
        )

        if not llm_results or len(llm_results[0]["boxes"]) == 0:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        input_boxes = llm_results[0]["boxes"].cpu().numpy()
        confidences = llm_results[0]["scores"].cpu().numpy()

        # Handle labels
        raw_labels = llm_results[0]["labels"]
        try:
            label_indices = raw_labels.cpu().numpy()
            class_names = [texts[0][int(li)] for li in label_indices]
            class_ids = label_indices.astype(int)
        except Exception:
            class_names = [str(l) for l in raw_labels]
            class_ids = np.arange(len(class_names))

        # Run SAM2 segmentation
        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": class_names,
            "class_ids": class_ids,
            "image_size": pil_image.size,
        }

    def _resolve_mask_overlaps(
        self, masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
    ) -> np.ndarray:
        """Resolve overlapping masks by keeping the highest confidence ones.

        Args:
            masks: Array of masks (N, H, W)
            scores: Array of confidence scores (N,)
            iou_threshold: IoU threshold above which masks are considered overlapping

        Returns:
            Array of indices for non-overlapping masks with highest confidence
        """
        n_masks = len(masks)
        if n_masks <= 1:
            return np.arange(n_masks)

        # Calculate IoU matrix
        intersections = np.zeros((n_masks, n_masks))
        unions = np.zeros((n_masks, n_masks))

        for i in range(n_masks):
            for j in range(i, n_masks):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                intersections[i, j] = intersections[j, i] = intersection
                unions[i, j] = unions[j, i] = union

        # Compute IoU matrix
        ious = intersections / (unions + 1e-6)

        # Keep track of which masks to keep
        keep = np.ones(n_masks, dtype=bool)

        # Sort by confidence (highest first)
        sorted_indices = np.argsort(scores)[::-1]

        for i, idx_i in enumerate(sorted_indices):
            if not keep[idx_i]:
                continue

            # Remove lower confidence masks that overlap significantly
            for j, idx_j in enumerate(sorted_indices[i + 1 :], i + 1):
                if keep[idx_j] and ious[idx_i, idx_j] > iou_threshold:
                    keep[idx_j] = False

        return np.where(keep)[0]

    def _predict_yolo_world(
        self,
        pil_image: Image.Image,
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Predict using YOLO-World + SAM2."""
        # Run YOLO-World detection
        results = self.yolo_world_model(pil_image, conf=box_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            # Return empty results if no detections
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        # Extract boxes and other info from YOLO results
        yolo_result = results[0]
        input_boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confidences = yolo_result.boxes.conf.cpu().numpy()
        yolo_class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)

        # Run SAM2 segmentation
        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        # get labels from YOLO class IDs
        labels = [self.yolo_world_model.names[int(cls_id)] for cls_id in yolo_class_ids]

        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": labels,
            "class_ids": yolo_class_ids,
            "image_size": pil_image.size,
        }

    def _predict_sam2_grid(
        self,
        pil_image: Image.Image,
        grid_size: int,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Predict using SAM2 with grid prompts to segment everything."""
        # Run SAM2 segmentation
        image = np.array(pil_image.convert("RGB"))
        masks = self.sam2_mask_generator.generate(image)

        if masks is None or len(masks) == 0:
            # Return empty results if no masks generated
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        # Extract boxes and scores from masks
        boxes = []
        valid_masks = []
        valid_scores = []
        for mask_dict in masks:
            # Each mask_dict contains:
            # - segmentation: the mask (H, W) bool array
            # - area: area in pixels
            # - bbox: [x, y, w, h] in XYWH format
            # - predicted_iou: model's prediction for mask quality
            # - point_coords: input point that generated this mask
            # - stability_score: additional mask quality measure
            # - crop_box: crop of the image used to generate this mask in XYWH format

            mask = mask_dict["segmentation"]
            bbox_xywh = mask_dict["bbox"]
            score = mask_dict.get("predicted_iou", 1.0)

            # Convert bbox from XYWH to XYXY
            x, y, w, h = bbox_xywh
            bbox_xyxy = [x, y, x + w, y + h]

            boxes.append(bbox_xyxy)
            valid_masks.append(mask)
            valid_scores.append(score)

        boxes = np.array(boxes)
        valid_masks = np.array(valid_masks)
        valid_scores = np.array(valid_scores)

        valid_masks = mask_subtract_contained(boxes, valid_masks.astype(bool))

        # Generate generic labels
        labels = [f"segment_{i}" for i in range(len(boxes))]
        class_ids = np.arange(len(boxes))

        return {
            "boxes": boxes,
            "masks": valid_masks.astype(bool),
            "scores": valid_scores,
            "labels": labels,
            "class_ids": class_ids,
            "image_size": pil_image.size,
        }

    def _predict_functional_elements(
        self, pil_image: Image.Image, box_threshold: float, multimask_output: bool
    ) -> Dict[str, Any]:
        """
        Predict functional elements in the image using RT-DETR model + SAM2.

        Args:
            pil_image: Input image as PIL Image
            box_threshold: Confidence threshold for bounding boxes
            multimask_output: Whether to return multiple masks per object

        Returns:
            Dictionary with keys:
            - 'boxes': Bounding boxes (N, 4) in xyxy format
            - 'masks': Segmentation masks (N, H, W)
            - 'scores': Confidence scores (N,)
            - 'labels': Class labels (N,)
            - 'class_ids': Class IDs (N,)
            - 'image_size': (width, height)
        """
        # Run functional element detection
        results = self.fungraph_model(pil_image, conf=box_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            # Return empty results if no detections
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        # Extract boxes and other info from RT-DETR results
        fungraph_result = results[0]
        input_boxes = fungraph_result.boxes.xyxy.cpu().numpy()
        confidences = fungraph_result.boxes.conf.cpu().numpy()
        fungraph_class_ids = fungraph_result.boxes.cls.cpu().numpy().astype(int)

        # Get class names from RT-DETR result
        labels = [
            self.fungraph_model.names[int(cls_id)] for cls_id in fungraph_class_ids
        ]

        # Run SAM2 segmentation
        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": labels,
            "class_ids": fungraph_class_ids,
            "image_size": pil_image.size,
        }

    def tag_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
    ) -> str:
        """
        Generate image tags using the selected backend.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            tags_backend: Optional override: 'ram' (default) or 'vlm'

        Returns:
            Tags as a single string in the format "tag1 | tag2 | tag3"
        """
        # Load and process image
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        # Ensure RAM is available
        self._ensure_ram_loaded()
        transformed_image = self.ram_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            res = inference(transformed_image, self.ram_model)
        return res[0]

    def visualize_results(
        self,
        results: Dict[str, Any],
        image: Union[str, np.ndarray],
        visualize: bool = True,
        output_path: Optional[str] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True,
        custom_color_map: Optional[List[str]] = None,
        apply_nms: bool = True,
        nms_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize detection and segmentation results.

        Args:
            results: Results from predict() method
            image: Original image (file path or numpy array)
            output_path: Path to save the visualized image
            show_boxes: Whether to draw bounding boxes
            show_masks: Whether to draw segmentation masks
            show_labels: Whether to draw labels
            custom_color_map: Custom color palette (list of hex colors)

        Returns:
            Annotated image as numpy array
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {image}")
        else:
            img = image.copy()

        # Check if there are any detections
        if len(results["boxes"]) == 0:
            if output_path:
                cv2.imwrite(output_path, img)
            return img

        # apply torchvision.ops.nms
        if apply_nms:
            boxes = torch.tensor(results["boxes"], dtype=torch.float32)
            scores = torch.tensor(results["scores"], dtype=torch.float32)
            keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_threshold)
            results["boxes"] = boxes[keep_indices].numpy()
            results["masks"] = results["masks"][keep_indices.numpy()]
            results["scores"] = scores[keep_indices].numpy()
            results["labels"] = [results["labels"][i] for i in keep_indices.numpy()]
            results["class_ids"] = results["class_ids"][keep_indices.numpy()]

        # Create detections object
        detections = sv.Detections(
            xyxy=results["boxes"], mask=results["masks"], class_id=results["class_ids"]
        )

        # Set up color palette
        if custom_color_map:
            color_palette = ColorPalette.from_hex(custom_color_map)
        else:
            color_palette = ColorPalette.DEFAULT

        annotated_frame = img.copy()

        # Draw boxes
        if show_boxes:
            box_annotator = sv.BoxAnnotator(color=color_palette)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        # Draw labels
        if show_labels:
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(results["labels"], results["scores"])
            ]
            label_annotator = sv.LabelAnnotator(color=color_palette)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        # Draw masks
        if show_masks:
            mask_annotator = sv.MaskAnnotator(color=color_palette)
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        elif visualize:
            # Convert to RGB for display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Annotated Image", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_frame

    def ram_tags_to_prompt(self, ram_tags: str) -> str:
        """
        Convert RAM tags string to GroundingSAM prompt format.

        Args:
            ram_tags: RAM tags string, e.g. "cat | car | dog"

        Returns:
            str: Prompt string, e.g. "cat. car. dog."
        """
        tags = [tag.strip() for tag in ram_tags.split("|") if tag.strip()]
        prompt = " ".join(f"{tag}." for tag in tags)
        return prompt

    def save_results_json(
        self, results: Dict[str, Any], image_path: str, output_path: str
    ):
        """
        Save detection results in JSON format.

        Args:
            results: Results from predict() method
            image_path: Path to the original image
            output_path: Path to save the JSON file
        """

        def single_mask_to_rle(mask):
            """Convert mask to RLE format."""
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8")
            )[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        # Convert masks to RLE format
        mask_rles = [single_mask_to_rle(mask) for mask in results["masks"]]

        # Prepare JSON data
        json_results = {
            "image_path": image_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "score": float(score),
                }
                for class_name, box, mask_rle, score in zip(
                    results["labels"], results["boxes"], mask_rles, results["scores"]
                )
            ],
            "box_format": "xyxy",
            "img_width": results["image_size"][0],
            "img_height": results["image_size"][1],
        }

        # Save JSON file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=4)


# Example usage
if __name__ == "__main__":
    # img_path = "/home/werby/datasets/ScanNetv2/scans/scene0000_00/color/2841.jpg"
    # img_path = "/home/werby/datasets/clio_data/apartment/images/rgb_560.jpg"
    # img_path = "/home/werby/datasets/rosbags/robot_data_2025-08-20-17-20-39_0/raw/rgb/rgb_image_1755710505974625024.jpg"
    img_path = "/home/werby/datasets/FunGraph3D/11bedroom/video1/rgb/frame_00529.jpg"
    # img_path = "/home/werby/Downloads/Media3.jpeg"
    # img_path = "/home/werby/datasets/Replica_RGBD/Replica/room0/results/frame000497.jpg"

    # Example 1: Grounding DINO instance (loads only GDINO + SAM2)
    print("=== Grounding DINO Mode ===")
    gsam_gdino = GroundingSAM2(
        detection_mode="grounding_dino",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id="IDEA-Research/grounding-dino-base",
        ram_pretrained="./checkpoints/ram_plus_swin_large_14m.pth",
        ram_image_size=384,
    )
    # Choose tagging backend: 'ram' (default) or 'vlm'
    eng_tags = gsam_gdino.tag_image(img_path, tags_backend="vlm")  # or 'ram'
    text_prompt = gsam_gdino.ram_tags_to_prompt(eng_tags)
    print(f"Text Prompt: {text_prompt}")
    results = gsam_gdino.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
        text_threshold=0.3,
    )
    gsam_gdino.visualize_results(
        results=results,
        image=img_path,
        output_path="output/annotated_image_grounding.jpg",
    )
    print(f"Found {len(results['labels'])} objects:")
    for label in results["labels"]:
        print(f" - {label}")

    del gsam_gdino

    # Example 2: YOLO-World instance (loads only YOLO-World + SAM2)
    print("\n=== YOLO-World Mode ===")
    gsam_yolo = GroundingSAM2(
        detection_mode="yolo_world",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        yolo_world_checkpoint="./checkpoints/yolov8x-worldv2.pt",
    )
    results_yolo = gsam_yolo.predict(
        image=img_path,
        box_threshold=0.3,
    )
    gsam_yolo.visualize_results(
        results=results_yolo,
        image=img_path,
        output_path="output/annotated_image_yolo.jpg",
    )
    print(f"Found {len(results_yolo['labels'])} objects:")
    for label in results_yolo["labels"]:
        print(f" - {label}")

    del gsam_yolo

    # Example 4: Functional Elements instance (loads only RT-DETR + SAM2)
    print("\n=== Functional Elements Mode ===")
    gsam_fun = GroundingSAM2(
        detection_mode="functional_elements",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        fungraph_checkpoint="./checkpoints/fungraph_det.pt",
    )
    results_fungraph = gsam_fun.predict(
        image=img_path,
        box_threshold=0.3,
    )
    gsam_fun.visualize_results(
        results=results_fungraph,
        image=img_path,
        output_path="output/annotated_image_fungraph.jpg",
    )
    print(f"Found {len(results_fungraph['labels'])} functional elements:")
    for label in results_fungraph["labels"]:
        print(f" - {label}")

    del gsam_fun

    # Example 5: LLMDet instance (loads only LLMDet + SAM2)
    print("\n=== LLMDet Mode ===")
    gsam_llmdet = GroundingSAM2(
        detection_mode="llmdet",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        llmdet_model_id="iSEE-Laboratory/llmdet_large",
    )
    eng_tags = gsam_llmdet.tag_image(img_path, tags_backend="vlm")  # or 'ram'
    print(f"English Tags: {eng_tags}")
    text_prompt = gsam_llmdet.ram_tags_to_prompt(eng_tags)
    results_llmdet = gsam_llmdet.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam_llmdet.visualize_results(
        results=results_llmdet,
        image=img_path,
        output_path="output/annotated_image_llmdet.jpg",
    )
    print(f"Found {len(results_llmdet['labels'])} objects:")
    for label in results_llmdet["labels"]:
        print(f" - {label}")

    del gsam_llmdet
    # Example 5: OwlV2 instance (loads only OwlV2 + SAM2)
    print("\n=== OwlV2 Mode ===")
    gsam_owl = GroundingSAM2(
        detection_mode="owlv2",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        owlv2_model_id="google/owlv2-base-patch16-ensemble",
    )
    eng_tags = gsam_owl.tag_image(img_path, tags_backend="vlm")  # or 'ram'
    print(f"English Tags: {eng_tags}")
    text_prompt = gsam_owl.ram_tags_to_prompt(eng_tags)
    results_owlv2 = gsam_owl.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam_owl.visualize_results(
        results=results_owlv2,
        image=img_path,
        output_path="output/annotated_image_owlv2.jpg",
    )
    print(f"Found {len(results_owlv2['labels'])} objects:")
    for label in results_owlv2["labels"]:
        print(f" - {label}")

    # Example 6: OwlV2 Functional Elements
    fun_tag = gsam_owl.tag_functional_elements(img_path)
    text_prompt = gsam_owl.ram_tags_to_prompt(fun_tag)
    results_owlv2 = gsam_owl.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam_owl.visualize_results(
        results=results_owlv2,
        image=img_path,
        output_path="output/annotated_image_fun_owlv2.jpg",
    )
    print(f"Found {len(results_owlv2['labels'])} objects:")
    for label in results_owlv2["labels"]:
        print(f" - {label}")

    del gsam_owl

    # Example 7: Dino functional elements
    print("\n=== Dino Functional Elements Mode ===")
    gsam_gdino = GroundingSAM2(
        detection_mode="grounding_dino",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id="IDEA-Research/grounding-dino-base",
        ram_pretrained="./checkpoints/ram_plus_swin_large_14m.pth",
        ram_image_size=384,
    )
    fun_tag = gsam_gdino.tag_functional_elements(img_path)
    text_prompt = gsam_gdino.ram_tags_to_prompt(fun_tag)
    results_dino = gsam_gdino.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.4,
    )
    gsam_gdino.visualize_results(
        results=results_dino,
        image=img_path,
        output_path="output/annotated_image_fun_dino.jpg",
    )
    print(f"Found {len(results_dino['labels'])} objects:")
    for label in results_dino["labels"]:
        print(f" - {label}")

    del gsam_gdino

    # Example 8: LLMDet Functional Elements
    gsam_llmdet = GroundingSAM2(
        detection_mode="llmdet",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        llmdet_model_id="iSEE-Laboratory/llmdet_large",
    )
    fun_tag = gsam_llmdet.tag_functional_elements(img_path)
    text_prompt = gsam_llmdet.ram_tags_to_prompt(fun_tag)
    results_llmdet = gsam_llmdet.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam_llmdet.visualize_results(
        results=results_llmdet,
        image=img_path,
        output_path="output/annotated_image_fun_llmdet.jpg",
    )
    print(f"Found {len(results_llmdet['labels'])} objects:")
    for label in results_llmdet["labels"]:
        print(f" - {label}")

    del gsam_llmdet
