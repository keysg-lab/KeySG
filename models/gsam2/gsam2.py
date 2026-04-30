import sys
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from typing import List, Dict, Any, Optional, Union
from supervision.draw.color import ColorPalette
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Make project root importable before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.llm.gpt_vlm import GPT_VLMInterface as VLMInterface
from keysg.utils.img_utils import mask_subtract_contained

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
    """SAM2 + LLMDet detection backend for object detection and segmentation."""

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_config: str,
        llmdet_model_id: Optional[str] = None,
        device: Optional[str] = None,
        force_cpu: bool = False,
        llmdet_max_tags_per_batch: int = 80,
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.llmdet_model_id = llmdet_model_id or "iSEE-Laboratory/llmdet_large"
        self.llmdet_max_tags_per_batch = llmdet_max_tags_per_batch
        self._vlm_client = None

        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
            )

        self._setup_environment()
        self._load_sam2()
        self._load_detection_model()

    def _setup_environment(self):
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _load_sam2(self):
        sam2_model = build_sam2(
            self.sam2_model_config, self.sam2_checkpoint, device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _load_detection_model(self):
        self.llmdet_processor = AutoProcessor.from_pretrained(self.llmdet_model_id)
        self.llmdet_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.llmdet_model_id
        ).to(self.device)

    def _ensure_vlm_client(self):
        if self._vlm_client is None:
            self._vlm_client = VLMInterface()

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold: float = 0.2,
        multimask_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect and segment objects using LLMDet + SAM2.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            text_prompt: Dot-separated object labels, e.g. "chair. table. door."
            box_threshold: Confidence threshold for bounding boxes
            multimask_output: Whether to return multiple masks per object

        Returns:
            dict with keys: 'boxes', 'masks', 'scores', 'labels', 'class_ids', 'image_size'
        """
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        self.sam2_predictor.set_image(np.array(pil_image.convert("RGB")))
        return self._predict_llmdet(
            pil_image, text_prompt, box_threshold, multimask_output
        )

    def _predict_llmdet(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Run LLMDet + SAM2 with automatic batching for large tag lists."""
        tags = [p.strip() for p in text_prompt.split(".") if p.strip()]

        if len(tags) <= self.llmdet_max_tags_per_batch:
            return self._predict_llmdet_batch(
                pil_image, [tags], box_threshold, multimask_output
            )

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

        merged_boxes = np.vstack(all_boxes)
        merged_masks = np.vstack(all_masks)
        merged_scores = np.hstack(all_scores)
        merged_class_ids = np.array(all_class_ids)

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
        inputs = self.llmdet_processor(
            images=pil_image,
            text=texts,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llmdet_model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)

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

        raw_labels = llm_results[0]["labels"]
        try:
            label_indices = raw_labels.cpu().numpy()
            class_names = [texts[0][int(li)] for li in label_indices]
            class_ids = label_indices.astype(int)
        except Exception:
            class_names = [str(label) for label in raw_labels]
            class_ids = np.arange(len(class_names))

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
        """Keep highest-confidence masks, suppressing overlapping ones above iou_threshold."""
        n_masks = len(masks)
        if n_masks <= 1:
            return np.arange(n_masks)

        intersections = np.zeros((n_masks, n_masks))
        unions = np.zeros((n_masks, n_masks))

        for i in range(n_masks):
            for j in range(i, n_masks):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                intersections[i, j] = intersections[j, i] = intersection
                unions[i, j] = unions[j, i] = union

        ious = intersections / (unions + 1e-6)
        keep = np.ones(n_masks, dtype=bool)
        sorted_indices = np.argsort(scores)[::-1]

        for i, idx_i in enumerate(sorted_indices):
            if not keep[idx_i]:
                continue
            for j, idx_j in enumerate(sorted_indices[i + 1 :], i + 1):
                if keep[idx_j] and ious[idx_i, idx_j] > iou_threshold:
                    keep[idx_j] = False

        return np.where(keep)[0]

    def tag_image(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Generate image tags using the VLM client.

        Returns:
            Tags as a single string in the format "tag1 | tag2 | tag3"
        """
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        self._ensure_vlm_client()
        tags = self._vlm_client.tag_objects_in_image(pil_image)
        return " | ".join(tags)

    def ram_tags_to_prompt(self, ram_tags: str) -> str:
        """Convert "tag1 | tag2 | tag3" to "tag1. tag2. tag3." for use as text_prompt."""
        tags = [tag.strip() for tag in ram_tags.split("|") if tag.strip()]
        return " ".join(f"{tag}." for tag in tags)

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
        """Visualize detection and segmentation results."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {image}")
        else:
            img = image.copy()

        if len(results["boxes"]) == 0:
            if output_path:
                cv2.imwrite(output_path, img)
            return img

        if apply_nms:
            boxes = torch.tensor(results["boxes"], dtype=torch.float32)
            scores = torch.tensor(results["scores"], dtype=torch.float32)
            keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_threshold)
            results["boxes"] = boxes[keep_indices].numpy()
            results["masks"] = results["masks"][keep_indices.numpy()]
            results["scores"] = scores[keep_indices].numpy()
            results["labels"] = [results["labels"][i] for i in keep_indices.numpy()]
            results["class_ids"] = results["class_ids"][keep_indices.numpy()]

        detections = sv.Detections(
            xyxy=results["boxes"], mask=results["masks"], class_id=results["class_ids"]
        )

        color_palette = (
            ColorPalette.from_hex(custom_color_map)
            if custom_color_map
            else ColorPalette.DEFAULT
        )
        annotated_frame = img.copy()

        if show_boxes:
            box_annotator = sv.BoxAnnotator(color=color_palette)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        if show_labels:
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(results["labels"], results["scores"])
            ]
            label_annotator = sv.LabelAnnotator(color=color_palette)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        if show_masks:
            mask_annotator = sv.MaskAnnotator(color=color_palette)
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_frame)
        elif visualize:
            cv2.imshow("Annotated Image", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_frame

    def save_results_json(
        self, results: Dict[str, Any], image_path: str, output_path: str
    ):
        """Save detection results to JSON with RLE-encoded masks."""

        def single_mask_to_rle(mask):
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8")
            )[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in results["masks"]]

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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=4)


# Example usage
if __name__ == "__main__":
    img_path = "/home/werby/SIRData/nico/raw/rosbag2_2026_03_26-07_39_28/rgb/rgb_image_1774510768749777920.jpg"
    # img_path = "/mnt/ssd2/datasets/SIR/Demo/fullab_1fps_da3/color/frame_0049.jpg"

    gsam = GroundingSAM2(
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        llmdet_model_id="iSEE-Laboratory/llmdet_large",
    )

    eng_tags = gsam.tag_image(img_path)
    print(f"Tags: {eng_tags}")

    text_prompt = gsam.ram_tags_to_prompt(eng_tags)
    results = gsam.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam.visualize_results(
        results=results,
        image=img_path,
        output_path="output/annotated_image_llmdet.jpg",
    )
    print(f"Found {len(results['labels'])} objects:")
    for label in results["labels"]:
        print(f" - {label}")
