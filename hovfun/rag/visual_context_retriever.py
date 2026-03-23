"""
VisualContextRetriever: Create embedding vector database for room frames using FAISS.

This class builds a vector database from room frame embeddings and performs
similarity matching with user queries to find the most relevant images.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from loguru import logger
from tqdm import tqdm

# Project imports - handle both installed package and direct execution scenarios
try:
    from hovfun.utils.clip_utils import CLIPFeatureExtractor
    from hovfun.scene_segmentor.scene_segmentor import SceneSegmentor
    from hovfun.scene_descriptor.scene_descriptor import SceneDescriptor
except ImportError:
    import sys
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from hovfun.utils.clip_utils import CLIPFeatureExtractor
    from hovfun.scene_segmentor.scene_segmentor import SceneSegmentor
    from hovfun.scene_descriptor.scene_descriptor import SceneDescriptor


@dataclass
class FrameData:
    """Data structure for frame information in the vector database."""

    room_id: str
    frame_index: int
    image_path: str
    tags: List[str]
    functional_tags: List[str]
    description: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result of visual context retrieval."""

    frame_data: FrameData
    similarity_score: float
    room_summary: Dict[str, Any]


class VisualContextRetriever:
    """
    Visual Context Retriever using FAISS for efficient similarity search.

    This class:
    1. Loads room data from segmentation and description outputs
    2. Creates CLIP embeddings for all room frames
    3. Builds a FAISS index for efficient similarity search
    4. Retrieves the most relevant frames for user queries
    """

    def __init__(
        self,
        output_dir: str,
        dataset: Any,
        clip_config: Optional[Dict[str, Any]] = None,
        faiss_index_type: str = "IndexFlatIP",  # Inner Product for cosine similarity
    ):
        """
        Initialize the VisualContextRetriever.

        Args:
            output_dir: Base output directory containing pipeline results
            dataset: Dataset instance for loading images
            clip_config: Configuration for CLIP feature extractor
            faiss_index_type: Type of FAISS index to use
        """
        self.output_dir = output_dir
        self.dataset = dataset
        self.clip_config = clip_config or {}
        self.faiss_index_type = faiss_index_type

        # Initialize CLIP extractor
        self.clip_extractor = CLIPFeatureExtractor(self.clip_config)

        # Storage for frames and embeddings
        self.frames_data: List[FrameData] = []
        self.frame_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.room_summaries: Dict[str, Dict[str, Any]] = {}

        # Paths for saving/loading
        self.index_cache_path = os.path.join(output_dir, "visual_index_cache.pkl")
        self.embeddings_cache_path = os.path.join(output_dir, "visual_embeddings.npy")
        self.faiss_index_path = os.path.join(output_dir, "visual_faiss.index")

    def load_room_data(self) -> None:
        """
        Load room data from segmentation and description outputs.
        Similar to the load_scene_segmentation and load_scene_description methods.
        """
        logger.info("Loading room data from {}", self.output_dir)

        # Load scene segmentation
        seg = SceneSegmentor(dataset=self.dataset, output_dir=self.output_dir)
        try:
            floors, floor_rooms = seg.load()
            dense_map, sampled_map = seg.get_room_pose_indices()

            self.floor_rooms = floor_rooms
            self.dense_map = dense_map
            self.sampled_map = sampled_map
            self.rooms = [room for _, rooms in floor_rooms for room in rooms]

        except FileNotFoundError as e:
            logger.error("Scene segmentation not found: {}", e)
            raise RuntimeError(
                f"Cannot load scene segmentation from {self.output_dir}. "
                f"Please run segmentation first."
            )

        # Load scene descriptions
        scene_descriptor = SceneDescriptor(
            dataset=self.dataset, output_dir=self.output_dir
        )
        try:
            room_descriptions = scene_descriptor.load_scene(self.output_dir)
            self.room_descriptions = room_descriptions
            logger.info(
                "Loaded scene descriptions for {} rooms", len(room_descriptions)
            )
        except FileNotFoundError as e:
            logger.warning("Scene descriptions not found: {}", e)
            self.room_descriptions = {}

    def _load_room_vlm_data(self, room_id: str, floor_id: str) -> Dict[str, Any]:
        """Load VLM data for a specific room."""
        room_dir = os.path.join(
            self.output_dir, "segmentation", f"floor_{floor_id}", f"room_{room_id}"
        )
        vlm_file = os.path.join(room_dir, f"room_{room_id}_vlm.json")

        if os.path.exists(vlm_file):
            with open(vlm_file, "r") as f:
                return json.load(f)
        else:
            logger.warning("VLM file not found for room {}: {}", room_id, vlm_file)
            return {}

    def build_frame_database(self, use_cache: bool = True) -> None:
        """
        Build the frame database from loaded room data.

        Args:
            use_cache: Whether to use cached data if available
        """
        if use_cache and os.path.exists(self.index_cache_path):
            logger.info("Loading cached frame database...")
            with open(self.index_cache_path, "rb") as f:
                cache_data = pickle.load(f)
                self.frames_data = cache_data["frames_data"]
                self.room_summaries = cache_data["room_summaries"]
            return

        logger.info("Building frame database...")
        self.frames_data = []
        self.room_summaries = {}

        # Process each room
        for floor, rooms in self.floor_rooms:
            for room in rooms:
                room_id = room.id
                floor_id = getattr(room, "floor_id", getattr(floor, "floor_id", "0"))

                # Load VLM data for this room
                vlm_data = self._load_room_vlm_data(room_id, floor_id)

                # Store room summary
                self.room_summaries[room_id] = {
                    "room_id": room_id,
                    "floor_id": floor_id,
                    "summary": vlm_data.get("summary", ""),
                    "total_frames": len(vlm_data.get("frames", [])),
                    "sparse_indices": getattr(room, "sparse_indices", []),
                }

                # Process each frame in the room
                for frame_info in vlm_data.get("frames", []):
                    frame_data = FrameData(
                        room_id=room_id,
                        frame_index=frame_info.get("index", -1),
                        image_path=frame_info.get("path", ""),
                        tags=frame_info.get("tags", []),
                        functional_tags=frame_info.get("functional_tags", []),
                        description=frame_info.get("description", {}),
                    )
                    self.frames_data.append(frame_data)

        logger.info(
            "Built frame database with {} frames from {} rooms",
            len(self.frames_data),
            len(self.room_summaries),
        )

        # Cache the data
        cache_data = {
            "frames_data": self.frames_data,
            "room_summaries": self.room_summaries,
        }
        with open(self.index_cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    def compute_embeddings(self, use_cache: bool = True, batch_size: int = 32) -> None:
        """
        Compute CLIP embeddings for all frames.

        Args:
            use_cache: Whether to use cached embeddings if available
            batch_size: Batch size for embedding computation
        """
        if use_cache and os.path.exists(self.embeddings_cache_path):
            logger.info("Loading cached embeddings...")
            self.frame_embeddings = np.load(self.embeddings_cache_path)
            return

        if not self.frames_data:
            raise RuntimeError(
                "Frame database not built. Call build_frame_database() first."
            )

        logger.info("Computing CLIP embeddings for {} frames...", len(self.frames_data))

        # Load images and compute embeddings in batches
        all_embeddings = []

        for i in tqdm(
            range(0, len(self.frames_data), batch_size), desc="Computing embeddings"
        ):
            batch_frames = self.frames_data[i : i + batch_size]
            batch_images = []

            for frame_data in batch_frames:
                try:
                    # Try to load from dataset first using frame index
                    if frame_data.frame_index >= 0:
                        rgb, _, _ = self.dataset[frame_data.frame_index]
                        img = (
                            Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
                        )
                    else:
                        # Fallback to loading from file path
                        img_path = frame_data.image_path
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(self.dataset.root_dir, img_path)
                        img = Image.open(img_path).convert("RGB")

                    batch_images.append(img)
                except Exception as e:
                    logger.warning(
                        "Failed to load image for frame {}: {}",
                        frame_data.frame_index,
                        e,
                    )
                    # Use a blank image as fallback
                    batch_images.append(Image.new("RGB", (224, 224), color="black"))

            # Compute embeddings for the batch
            if batch_images:
                batch_embeddings = self.clip_extractor.get_img_feats_batch(
                    batch_images, batch_size
                )
                all_embeddings.append(batch_embeddings)

        # Combine all embeddings
        if all_embeddings:
            self.frame_embeddings = np.vstack(all_embeddings).astype(np.float32)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.frame_embeddings)

            # Cache the embeddings
            np.save(self.embeddings_cache_path, self.frame_embeddings)

            logger.info("Computed embeddings shape: {}", self.frame_embeddings.shape)
        else:
            raise RuntimeError("No embeddings computed")

    def build_faiss_index(self, use_cache: bool = True) -> None:
        """
        Build FAISS index for efficient similarity search.

        Args:
            use_cache: Whether to use cached index if available
        """
        if use_cache and os.path.exists(self.faiss_index_path):
            logger.info("Loading cached FAISS index...")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            return

        if self.frame_embeddings is None:
            raise RuntimeError(
                "Embeddings not computed. Call compute_embeddings() first."
            )

        logger.info("Building FAISS index...")

        # Create FAISS index
        embedding_dim = self.frame_embeddings.shape[1]

        if self.faiss_index_type == "IndexFlatIP":
            # Inner Product index for cosine similarity (since embeddings are normalized)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        elif self.faiss_index_type == "IndexFlatL2":
            # L2 distance index
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")

        # Add embeddings to index
        self.faiss_index.add(self.frame_embeddings)

        # Cache the index
        faiss.write_index(self.faiss_index, self.faiss_index_path)

        logger.info("Built FAISS index with {} vectors", self.faiss_index.ntotal)

    def setup(self, use_cache: bool = True, batch_size: int = 32) -> None:
        """
        Complete setup: load data, build database, compute embeddings, and build index.

        Args:
            use_cache: Whether to use cached data if available
            batch_size: Batch size for embedding computation
        """
        self.load_room_data()
        self.build_frame_database(use_cache=use_cache)
        self.compute_embeddings(use_cache=use_cache, batch_size=batch_size)
        self.build_faiss_index(use_cache=use_cache)

    def retrieve_similar_frames(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
        include_room_context: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve frames most similar to the query.

        Args:
            query: Text query or list of query nouns
            top_k: Number of top results to return
            include_room_context: Whether to include room summary in results

        Returns:
            List of RetrievalResult objects
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call setup() first.")

        # Convert query to text if it's a list
        if isinstance(query, list):
            query_text = " ".join(query)
        else:
            query_text = query

        logger.info("Retrieving frames for query: '{}'", query_text)

        # Compute query embedding
        query_embedding = self.clip_extractor.get_text_feats([query_text])

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        # Create results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.frames_data):
                continue

            frame_data = self.frames_data[idx]

            # Get room summary if requested
            room_summary = {}
            if include_room_context and frame_data.room_id in self.room_summaries:
                room_summary = self.room_summaries[frame_data.room_id].copy()

            result = RetrievalResult(
                frame_data=frame_data,
                similarity_score=float(score),
                room_summary=room_summary,
            )
            results.append(result)

        logger.info("Retrieved {} results", len(results))
        return results

    def retrieve_with_noun_matching(
        self,
        query_nouns: List[str],
        top_k: int = 5,
        semantic_weight: float = 0.7,
        tag_weight: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve frames using both semantic similarity and tag matching.

        Args:
            query_nouns: List of query nouns
            top_k: Number of top results to return
            semantic_weight: Weight for semantic similarity
            tag_weight: Weight for tag matching

        Returns:
            List of RetrievalResult objects
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call setup() first.")

        logger.info("Retrieving frames for nouns: {}", query_nouns)

        # Get semantic similarity scores
        query_text = " ".join(query_nouns)
        query_embedding = self.clip_extractor.get_text_feats([query_text])
        faiss.normalize_L2(query_embedding)

        # Get many more candidates for reranking
        candidate_k = min(top_k * 10, len(self.frames_data))
        semantic_scores, semantic_indices = self.faiss_index.search(
            query_embedding, candidate_k
        )

        # Compute tag matching scores
        final_scores = []
        query_nouns_lower = [noun.lower() for noun in query_nouns]

        for semantic_score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx >= len(self.frames_data):
                continue

            frame_data = self.frames_data[idx]

            # Compute tag matching score
            all_tags = frame_data.tags + frame_data.functional_tags
            all_tags_lower = [tag.lower() for tag in all_tags]

            # Count matches
            tag_matches = sum(1 for noun in query_nouns_lower if noun in all_tags_lower)
            tag_score = tag_matches / len(query_nouns_lower) if query_nouns_lower else 0

            # Combine scores
            combined_score = semantic_weight * semantic_score + tag_weight * tag_score

            final_scores.append((combined_score, idx, semantic_score, tag_score))

        # Sort by combined score and take top-k
        final_scores.sort(reverse=True, key=lambda x: x[0])
        final_scores = final_scores[:top_k]

        # Create results
        results = []
        for combined_score, idx, semantic_score, tag_score in final_scores:
            frame_data = self.frames_data[idx]

            # Get room summary
            room_summary = {}
            if frame_data.room_id in self.room_summaries:
                room_summary = self.room_summaries[frame_data.room_id].copy()
                # Add scoring details
                room_summary["retrieval_scores"] = {
                    "combined": float(combined_score),
                    "semantic": float(semantic_score),
                    "tag_matching": float(tag_score),
                }

            result = RetrievalResult(
                frame_data=frame_data,
                similarity_score=float(combined_score),
                room_summary=room_summary,
            )
            results.append(result)

        logger.info("Retrieved {} results with combined scoring", len(results))
        return results

    def get_frame_image(self, frame_data: FrameData) -> Image.Image:
        """
        Load the actual image for a frame.

        Args:
            frame_data: Frame data containing image information

        Returns:
            PIL Image
        """
        try:
            # Try to load from dataset first using frame index
            if frame_data.frame_index >= 0:
                rgb, _, _ = self.dataset[frame_data.frame_index]
                return Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
            else:
                # Fallback to loading from file path
                img_path = frame_data.image_path
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.dataset.root_dir, img_path)
                return Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(
                "Failed to load image for frame {}: {}", frame_data.frame_index, e
            )
            # Return a blank image as fallback
            return Image.new("RGB", (224, 224), color="black")

    def format_context_for_llm(
        self, results: List[RetrievalResult], include_images: bool = False
    ) -> Dict[str, Any]:
        """
        Format retrieval results for LLM context.

        Args:
            results: List of retrieval results
            include_images: Whether to include image data (base64 encoded)

        Returns:
            Formatted context dictionary
        """
        context = {
            "query_results": [],
            "room_summaries": {},
            "total_results": len(results),
        }

        for result in results:
            frame_info = {
                "room_id": result.frame_data.room_id,
                "frame_index": result.frame_data.frame_index,
                "similarity_score": result.similarity_score,
                "tags": result.frame_data.tags,
                "functional_tags": result.frame_data.functional_tags,
                "description": result.frame_data.description,
                "image_path": result.frame_data.image_path,
            }

            if include_images:
                try:
                    img = self.get_frame_image(result.frame_data)
                    # Convert to base64 for LLM
                    import base64
                    import io

                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    frame_info["image_base64"] = img_base64
                except Exception as e:
                    logger.warning(
                        "Failed to encode image for frame {}: {}",
                        result.frame_data.frame_index,
                        e,
                    )

            context["query_results"].append(frame_info)

            # Add room summary (avoid duplicates)
            if result.frame_data.room_id not in context["room_summaries"]:
                context["room_summaries"][
                    result.frame_data.room_id
                ] = result.room_summary

        return context


# Example usage
if __name__ == "__main__":
    # This would typically be used within the main pipeline

    # Example configuration
    output_dir = "output/pipeline/HMP3D/00824-Dd4bFSTQ8gi"

    # This would be your actual dataset instance
    # dataset = HM3DSemDataset(config)

    # Initialize retriever
    # retriever = VisualContextRetriever(
    #     output_dir=output_dir,
    #     dataset=dataset,
    #     clip_config={"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}
    # )

    # Setup the retriever
    # retriever.setup(use_cache=True)

    # Example queries
    # results = retriever.retrieve_similar_frames("bedroom with bed and lamp", top_k=3)
    # results = retriever.retrieve_with_noun_matching(["bed", "lamp", "nightstand"], top_k=5)

    # Format for LLM
    # llm_context = retriever.format_context_for_llm(results, include_images=True)

    print("VisualContextRetriever example completed!")
