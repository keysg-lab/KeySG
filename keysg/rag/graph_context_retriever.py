"""GraphContextRetriever: hierarchical (floor/room/frame/object) Graph RAG.

Refactored for brevity: most pure helper logic lives in `graph_rag_utils`.
This class now orchestrates loading, embedding, indexing, and searching.
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any, Optional, Sequence, Tuple, Set
from dataclasses import dataclass
from pydantic import BaseModel, Field

import numpy as np
from loguru import logger
from tqdm import tqdm as _tqdm
import faiss


from .graph_rag_utils import Chunk
from .graph_rag_utils import (
    load_scene_description,
    build_chunks_from_descriptions,
    normalize_embeddings,
    prepare_text_query,
    combine_search_results,
    gather_object_chunks,
    ensure_clip as _ensure_clip_util,
    compute_frame_visual_embeddings as _compute_frame_visual_embeddings_util,
    build_frame_visual_faiss_index as _build_frame_visual_faiss_index_util,
)

# Import utilities - handle both installed package and direct execution scenarios
try:
    from keysg.utils.load_utils import load_scene_nodes, get_objects
except ImportError:
    import sys
    import os

    # Fallback for when running from within the package
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from keysg.utils.load_utils import load_scene_nodes, get_objects


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class GraphContextRetriever:
    def __init__(self, output_dir: str):
        # Base directory
        self.output_dir = output_dir.rstrip("/")
        # Scene description data
        self.floor_summaries: Dict[str, Any] = {}
        self.room_vlm: Dict[str, Any] = {}
        self.room_index_paths: Dict[str, Any] = {}
        self.chunks: List[Chunk] = []

        # Embeddings / FAISS
        self.embeddings: Optional[np.ndarray] = None
        self.embed_model_name: Optional[str] = None
        self.index = None  # text FAISS index

        # LLM + CLIP (lazy init)
        self.gpt = None
        self.clip = None
        self.clip_model_id: Optional[str] = None

        # Frame visual indexing
        self.frame_visual_embeddings: Optional[np.ndarray] = None
        self.frame_visual_index = None
        self.frame_visual_chunk_indices: List[int] = []
        # Object visual (CLIP) features indexing (from object.feature vectors)
        self.object_visual_embeddings: Optional[np.ndarray] = None
        self.object_visual_index = None
        self.object_visual_chunk_indices: List[int] = []

        # Cache artifact paths (still used for text & frame visual; object caching removed)
        self.output_dir = output_dir
        self.rag_save_path = os.path.join(self.output_dir, "rag_cache")
        os.makedirs(self.rag_save_path, exist_ok=True)
        self.meta_path = os.path.join(self.rag_save_path, "graph_chunks_meta.json")
        self.emb_path = os.path.join(self.rag_save_path, "graph_embeddings.npy")
        self.index_path = os.path.join(self.rag_save_path, "graph_faiss.index")
        self.frame_vis_emb_path = os.path.join(
            self.rag_save_path, "graph_frame_visual_embeddings.npy"
        )
        self.frame_vis_index_path = os.path.join(
            self.rag_save_path, "graph_frame_visual_faiss.index"
        )
        self.frame_vis_meta_path = os.path.join(
            self.rag_save_path, "graph_frame_visual_meta.json"
        )
        # Object visual cache artifact paths
        self.object_vis_emb_path = os.path.join(
            self.rag_save_path, "graph_object_visual_embeddings.npy"
        )
        self.object_vis_index_path = os.path.join(
            self.rag_save_path, "graph_object_visual_faiss.index"
        )
        self.object_vis_meta_path = os.path.join(
            self.rag_save_path, "graph_object_visual_meta.json"
        )

    # ------------------------------------------------------------------
    # Loading description artifacts
    # ------------------------------------------------------------------
    def load_description_files(self) -> None:
        """Load scene description JSONs (idempotent)."""
        if self.floor_summaries or self.room_vlm:
            logger.info("Scene description files already loaded")
            return
        (self.floor_summaries, self.room_vlm, self.room_index_paths) = (
            load_scene_description(self.output_dir)
        )

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------
    def build_chunks(self) -> List[Chunk]:
        if self.chunks:
            return self.chunks
        self.load_description_files()
        self.chunks = build_chunks_from_descriptions(
            self.floor_summaries,
            self.room_vlm,
            self.room_index_paths,
            output_dir=self.output_dir,
        )
        logger.info("Built {} chunks", len(self.chunks))
        return self.chunks

    def compute_embeddings(
        self,
        model_name: str = "text-embedding-3-small",
        use_cache: bool = True,
        compute_frame_visual: bool = True,
        compute_object_visual: bool = True,
        clip_config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Compute (or load cached) text embeddings (OpenAI) and optionally frame visual embeddings."""
        if not self.chunks:
            self.build_chunks()

        if use_cache and self._load_cached_embeddings(model_name):
            if compute_frame_visual:
                self._ensure_frame_visual_embeddings(clip_config)
            if compute_object_visual:
                self._ensure_object_visual_embeddings()
            return self.embeddings  # type: ignore

        # Embedding models have a per-input token limit (e.g. 8192 for
        # text-embedding-3-small).  Truncate overly long texts to stay under
        # the limit (~4 chars/token heuristic, with safety margin).
        _MAX_EMBED_CHARS = 30_000  # ~7500 tokens, safely under 8192
        texts = []
        truncated = 0
        for c in self.chunks:
            t = c.content
            if len(t) > _MAX_EMBED_CHARS:
                t = t[:_MAX_EMBED_CHARS]
                truncated += 1
            texts.append(t)
        if truncated:
            logger.info("Truncated {} chunks to {} chars for embedding", truncated, _MAX_EMBED_CHARS)
        logger.info("Embedding {} chunks (model={})", len(texts), model_name)
        self._ensure_gpt()
        with _tqdm(
            total=len(texts), desc="Text embeddings (API batch)", unit="chunk"
        ) as pbar:
            raw = self.gpt.embed_text(texts, model=model_name)  # type: ignore[attr-defined]
            pbar.update(len(texts))
        self.embeddings = normalize_embeddings(np.asarray(raw, dtype="float32"))
        self.embed_model_name = model_name
        np.save(self.emb_path, self.embeddings)
        self.save_metadata()
        if compute_frame_visual:
            self._ensure_frame_visual_embeddings(clip_config)
        if compute_object_visual:
            self._ensure_object_visual_embeddings()
        return self.embeddings  # type: ignore

    def _load_cached_embeddings(self, model_name: str) -> bool:
        """Load cached embeddings if available, model matches, and chunk count matches."""
        if not (os.path.exists(self.emb_path) and os.path.exists(self.meta_path)):
            return False
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("embedding_model") != model_name:
                return False
            # Validate chunk count from meta (fast path)
            cached_count = meta.get("total_chunks")
            if cached_count is not None and cached_count != len(self.chunks):
                logger.info(
                    "Cached embeddings chunk count mismatch ({} vs {}); recomputing.",
                    cached_count,
                    len(self.chunks),
                )
                return False
            emb = np.load(self.emb_path)
            # Double-check embedding row count (guards against meta/npy desync)
            if emb.shape[0] != len(self.chunks):
                logger.info(
                    "Cached embeddings row count mismatch ({} vs {}); recomputing.",
                    emb.shape[0],
                    len(self.chunks),
                )
                return False
            self.embeddings = emb
            self.embed_model_name = model_name
            logger.info(
                "Loaded cached embeddings with shape {} (model: {})",
                self.embeddings.shape,
                model_name,
            )
            return True
        except Exception:
            pass
        return False

    def _ensure_frame_visual_embeddings(self, clip_config: Optional[Dict[str, Any]]):
        """Compute or load frame visual embeddings if needed."""
        try:
            _compute_frame_visual_embeddings_util(
                self, clip_config=clip_config, use_cache=True
            )
            _build_frame_visual_faiss_index_util(self, use_cache=True)
        except Exception as e:
            logger.warning("Failed to compute/load frame visual embeddings: {}", e)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def build_faiss_index(self, use_cache: bool = True):
        if faiss is None:
            raise RuntimeError("faiss not installed; install faiss-cpu or faiss-gpu")
        if self.embeddings is None:
            self.compute_embeddings()
        assert self.embeddings is not None
        d = self.embeddings.shape[1]
        if use_cache and os.path.exists(self.index_path):
            try:
                idx = faiss.read_index(self.index_path)
                if idx.ntotal == len(self.chunks):
                    self.index = idx
                    logger.info("Loaded cached FAISS index ({} vectors)", idx.ntotal)
                    return
            except Exception:
                pass
        logger.info("Building FAISS text index (dim={})", d)
        index = faiss.IndexFlatIP(d)
        index.add(self.embeddings)
        faiss.write_index(index, self.index_path)
        self.index = index

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------
    def save_metadata(self):
        records = [c.to_record() for c in self.chunks]
        payload = {
            "version": 1,
            "created": time.time(),
            "total_chunks": len(records),
            "embedding_model": self.embed_model_name,
            "embedding_backend": "openai",
            "chunks": records,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved chunk metadata to {}", self.meta_path)

    def _ensure_gpt(self):
        """Lazy-load GPTInterface."""
        if self.gpt is None:
            try:
                from models.llm.openai_api import GPTInterface
            except ImportError:
                raise RuntimeError(
                    "GPTInterface not available. Ensure OpenAI dependencies are installed."
                )
            self.gpt = GPTInterface()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_types: Optional[Sequence[str]] = None,
        expand_factor: int = 10,
        frame_modality: str = "both",  # one of {"text","visual","both"}
        object_modality: str = "both",  # one of {"text","visual","both"} for object chunks
        clip_config: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Semantic search over indexed chunks.

        Args:
            query: User text query
            top_k: Desired number of returned results
            doc_types: Optional subset filter (e.g. ["object", "frame"]) if provided we retrieve
                       more candidates then filter
            expand_factor: When filtering by doc_types, multiply top_k by this to get enough candidates
            frame_modality: For frame doc_type choose similarity modality: 'text', 'visual', or 'both'.
            object_modality: For object doc_type choose similarity modality: 'text', 'visual', or 'both'.
            clip_config: Optional dict to override default CLIP model config.
        """
        frame_modality = frame_modality.lower().strip()
        if frame_modality not in {"text", "visual", "both"}:
            raise ValueError("frame_modality must be one of {'text','visual','both'}")
        object_modality = object_modality.lower().strip()
        if object_modality not in {"text", "visual", "both"}:
            raise ValueError("object_modality must be one of {'text','visual','both'}")

        doc_type_set: Optional[Set[str]] = (
            set(dt.lower() for dt in doc_types) if doc_types else None
        )

        # Determine if text index needed (any doc type requiring text embedding)
        searching_frames = not doc_type_set or "frame" in doc_type_set
        searching_objects = not doc_type_set or "object" in doc_type_set
        need_text = (
            (frame_modality in {"text", "both"} and searching_frames)
            or (object_modality in {"text", "both"} and searching_objects)
            or (
                doc_type_set
                and any(dt not in {"frame", "object"} for dt in doc_type_set)
            )
        )

        if need_text and (self.index is None or self.embeddings is None):
            raise RuntimeError(
                "Text search requested but embeddings or index are not loaded. Please run compute_embeddings() and build_faiss_index() first."
            )

        if frame_modality in {"visual", "both"}:
            if (
                self.frame_visual_index is None
                or self.frame_visual_embeddings is None
                or self.frame_visual_embeddings.size == 0
            ):
                raise RuntimeError(
                    "Frame visual search requested but visual embeddings/index missing. Run compute_embeddings(compute_frame_visual=True) first."
                )

        # Ensure object visual embeddings if requested
        if object_modality in {"visual", "both"}:
            try:
                self._ensure_object_visual_embeddings()
            except Exception as _ove:
                logger.warning(
                    "Object visual embedding init failed ({}); degrading to text-only for objects.",
                    _ove,
                )
                object_modality = "text"
            else:
                if (
                    self.object_visual_index is None
                    or self.object_visual_embeddings is None
                    or self.object_visual_embeddings.size == 0
                ):
                    logger.warning(
                        "Object visual index unavailable; degrading to text-only for objects."
                    )
                    object_modality = "text"

        # expanded top-K for initial search if filtering by doc type
        k_search = (
            top_k
            if not doc_type_set
            else min(len(self.chunks), max(top_k * expand_factor, top_k))
        )

        # Text search
        text_results: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if need_text:
            self._ensure_gpt()
            q_emb = prepare_text_query(
                self.gpt, query, self.embed_model_name or "text-embedding-3-small"
            )
            text_results = self.index.search(q_emb, k_search)  # type: ignore

        # Visual search (frames & objects) via CLIP text query embedding (compute once if needed)
        frame_visual_results: Optional[Tuple[np.ndarray, np.ndarray]] = None
        object_visual_results: Optional[Tuple[np.ndarray, np.ndarray]] = None
        need_clip_text = frame_modality in {"visual", "both"} or object_modality in {
            "visual",
            "both",
        }
        clip_text_query = None
        if need_clip_text:
            _ensure_clip_util(self, clip_config)
            clip_text_query = self.clip.get_text_feats([query]).astype("float32")  # type: ignore[attr-defined]
            clip_text_query /= (
                np.linalg.norm(clip_text_query, axis=1, keepdims=True) + 1e-9
            )
        # Frames
        if clip_text_query is not None and frame_modality in {"visual", "both"}:
            k_frames = min(
                len(self.frame_visual_chunk_indices),
                max(k_search, len(self.frame_visual_chunk_indices)),
            )
            frame_visual_results = self.frame_visual_index.search(clip_text_query, k_frames)  # type: ignore
        # Objects
        if clip_text_query is not None and object_modality in {"visual", "both"}:
            k_objs = min(
                len(self.object_visual_chunk_indices),
                max(k_search, len(self.object_visual_chunk_indices)),
            )
            if k_objs > 0:
                object_visual_results = self.object_visual_index.search(clip_text_query, k_objs)  # type: ignore

        matches = combine_search_results(
            chunks=self.chunks,
            text_results=text_results,
            visual_results=frame_visual_results,
            frame_visual_chunk_indices=self.frame_visual_chunk_indices,
            object_visual_results=object_visual_results,
            object_visual_chunk_indices=self.object_visual_chunk_indices,
            doc_type_set=doc_type_set,
            top_k=top_k,
        )
        results = {}
        for mod, res in matches.items():
            # logger.info("Search modality '{}' returned {} results", mod, len(res))
            results[mod] = [
                SearchResult(chunk=self.chunks[idx], score=score) for idx, score in res
            ]
        return results

    # ------------------------------------------------------------------
    # Object visual feature handling
    # ------------------------------------------------------------------
    def _ensure_object_visual_embeddings(self, use_cache: bool = True) -> None:
        if (
            self.object_visual_embeddings is not None
            and self.object_visual_index is not None
        ):
            return

        # Try loading from disk cache
        if use_cache and (
            os.path.exists(self.object_vis_emb_path)
            and os.path.exists(self.object_vis_meta_path)
            and os.path.exists(self.object_vis_index_path)
        ):
            try:
                with open(self.object_vis_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                chunks_ok = meta.get("total_chunks", -1) == len(self.chunks)
                emb = np.load(self.object_vis_emb_path)
                indices = meta.get("object_chunk_indices", [])
                shape_ok = bool(indices) and emb.shape[0] == len(indices)
                if chunks_ok and shape_ok:
                    self.object_visual_embeddings = emb
                    self.object_visual_chunk_indices = indices
                    self.object_visual_index = faiss.read_index(
                        self.object_vis_index_path
                    )
                    logger.info(
                        "Loaded cached object visual FAISS index ({} vectors)",
                        self.object_visual_index.ntotal,
                    )
                    return
                logger.info(
                    "Object visual cache invalid (chunks_ok=%s shape_ok=%s); rebuilding",
                    chunks_ok,
                    shape_ok,
                )
            except Exception as e:
                logger.warning("Failed to load cached object visual embeddings: {}", e)

        self.nodes = load_scene_nodes(self.output_dir)
        self.objects = get_objects(self.nodes)
        if not self.objects:
            logger.warning(
                "No objects found in scene segmentation; skipping object visual index"
            )
            self.object_visual_embeddings = np.zeros((0, 512), dtype="float32")
            self.object_visual_index = None
            self.object_visual_chunk_indices = []
            return

        # Build O(1) lookup dict (was O(N) linear scan per object = O(N²) total)
        obj_by_id = {o.id: o for o in self.objects}

        self.object_chunks = gather_object_chunks(self.chunks)
        # First pass: collect features and detect the dominant dimensionality
        raw_feats: list = []
        raw_indices: list = []
        for idx, c in _tqdm(
            self.object_chunks,
            desc="Building object visual index",
            unit="obj",
            leave=False,
        ):
            obj_id = c.metadata.get("object_id")
            obj_node = obj_by_id.get(obj_id)
            if obj_node is not None and obj_node.feature is not None:
                feat = obj_node.feature.astype("float32").ravel()
                feat /= np.linalg.norm(feat) + 1e-9
                raw_feats.append(feat)
                raw_indices.append(idx)

        # Determine the most common feature dimension and keep only matching ones
        obj_nodes_feats = []
        if raw_feats:
            from collections import Counter

            dim_counts = Counter(f.shape[0] for f in raw_feats)
            target_dim = dim_counts.most_common(1)[0][0]
            skipped = 0
            for feat, idx in zip(raw_feats, raw_indices):
                if feat.shape[0] == target_dim:
                    obj_nodes_feats.append(feat)
                    self.object_visual_chunk_indices.append(idx)
                else:
                    skipped += 1
            if skipped:
                logger.info(
                    "Skipped {} objects with mismatched feature dim (expected {})",
                    skipped,
                    target_dim,
                )

        if not obj_nodes_feats:
            logger.warning(
                "No valid object visual features found; object visual index not built."
            )
            sample_dim = next(
                (o.feature.shape[0] for o in self.objects if o.feature is not None), 512
            )
            self.object_visual_embeddings = np.zeros((0, sample_dim), dtype="float32")
            self.object_visual_index = None
            self.object_visual_chunk_indices = []
            return

        self.object_visual_embeddings = np.stack(obj_nodes_feats, axis=0)
        d = self.object_visual_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(self.object_visual_embeddings)
        self.object_visual_index = index
        logger.info("Built object visual FAISS index ({} vectors)", index.ntotal)

        # Persist to disk cache
        try:
            np.save(self.object_vis_emb_path, self.object_visual_embeddings)
            faiss.write_index(index, self.object_vis_index_path)
            with open(self.object_vis_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "created": time.time(),
                        "n_objects": len(self.object_visual_chunk_indices),
                        "total_chunks": len(self.chunks),
                        "object_chunk_indices": self.object_visual_chunk_indices,
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "Saved object visual cache ({} objects)",
                len(self.object_visual_chunk_indices),
            )
        except Exception as e:
            logger.warning("Failed to save object visual cache: {}", e)

    # ------------------------------------------------------------------
    # LLM Answer Generation from Context Package
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        context_package: Dict[str, Any],
        *,
        model: str = "gpt-5-mini",
        structured: bool = True,
        return_text_only: bool = False,
    ) -> Any:
        """Generate a (structured) answer using the assembled context package.

        Returns (default) a dict with keys:
            answer: str (natural language answer)
            intent: Optional[str] (if provided in context_package['query_analysis'])
            room_id: Optional[str] (primary room – only when a single room focus)
            object_id: Optional[str] (primary object – only when a single object focus)
            rooms: List[str] (all referenced / candidate rooms relevant to the answer)
            objects: List[str] (all referenced / candidate objects relevant to the answer)

        When the inferred / provided intent indicates planning or multiple objects/rooms
        (plan_multi_object_task | multi_level_object_retrieval | locate_multiple_objects)
        the model is encouraged to fill the lists `rooms` / `objects` (ordered if path / plan).

        Args:
            context_package: Dict with at least keys 'query' and 'context'. Optionally includes 'query_analysis'.
            model: LLM model name.
            structured: If True (default) try structured JSON via GPT structured output.
            return_text_only: If True, only return the answer string (compat mode).

        Fallback: if structured call fails, falls back to plain text answer and heuristic id extraction.
        """
        self._ensure_gpt()
        query = context_package.get("query")
        ctx = context_package.get("context", {}) or {}
        qa = context_package.get("query_analysis") or {}
        intent = qa.get("intent") if isinstance(qa, dict) else None

        # Collect candidate ids from context
        room_record = ctx.get("room") or {}
        primary_room_id = (
            room_record.get("metadata", {}).get("room_id")
            or room_record.get("metadata", {}).get("room")
            or room_record.get("id")
        )
        object_records = ctx.get("objects") or []
        frame_records = ctx.get("frames") or []

        candidate_object_ids: List[str] = []
        for o in object_records:
            oid = (
                o.get("metadata", {}).get("object_id")
                or o.get("metadata", {}).get("id")
                or o.get("id")
            )
            if oid and oid not in candidate_object_ids:
                candidate_object_ids.append(oid)

        candidate_room_ids = [r for r in [primary_room_id] if r]

        # Formatting helper for prompt context
        def _fmt_block(name: str, items: Any) -> str:
            if not items:
                return f"{name}: <none>"
            if isinstance(items, list):
                return (
                    name
                    + ":\n"
                    + "\n".join(
                        f"- id={it.get('id')} score={it.get('score')} :: {it.get('content', '')[:220]}"
                        for it in items
                    )
                )
            elif isinstance(items, dict):
                return (
                    name + f": id={items.get('id')} :: {items.get('content','')[:260]}"
                )
            return name + ": <unhandled>"

        # Instruction tuning depending on intent
        multi_intents = {
            "plan_multi_object_task",
            "multi_level_object_retrieval",
            "locate_multiple_objects",
        }
        is_multi = intent in multi_intents

        # Structured schema --------------------------------------------------
        class _AnswerSchema(BaseModel):  # type: ignore[misc]
            answer: str = Field(description="Concise grounded answer. 1-4 sentences.")
            intent: Optional[str] = Field(
                default=None, description="Echo of interpreted intent or None"
            )
            room_id: Optional[str] = Field(
                default=None,
                description="Primary room id if a single dominant room is referenced",
            )
            object_id: Optional[str] = Field(
                default=None,
                description="Primary object id if a single dominant object is referenced",
            )
            rooms: List[str] = Field(
                default_factory=list,
                description="List (ordered if path/plan) of relevant room ids from provided context only.",
            )
            objects: List[str] = Field(
                default_factory=list,
                description="List of relevant object ids from provided context only.",
            )

        base_instructions = [
            "You are an indoor scene assistant generating structured JSON.",
            "Use ONLY provided context objects / rooms / frames.",
            "Never invent ids not present.",
            "If unsure, leave fields null or empty list; never hallucinate.",
            "Lists must preserve logical order if implying a path or sequence.",
            "Answer must remain concise and grounded (no speculation).",
        ]
        if is_multi:
            base_instructions.append(
                "Intent suggests multi-room/object plan; include ordered 'rooms' / 'objects' lists when applicable."
            )
        else:
            base_instructions.append(
                "If only one salient room/object, set primary room_id/object_id; keep lists either empty or containing that id if helpful."
            )

        instructions = " ".join(base_instructions)

        # Provide explicit candidate ids to constrain model
        candidate_summary = (
            f"Candidate room ids: {candidate_room_ids or ['<none>']}\n"
            f"Candidate object ids: {candidate_object_ids or ['<none>']}\n"
        )

        prompt = (
            f"User Query: {query}\n\n"
            + candidate_summary
            + _fmt_block("Room", room_record)
            + "\n"
            + _fmt_block("Objects", object_records)
            + "\n"
            + _fmt_block("Frames", frame_records)
        )

        # Attempt structured prompt
        if structured:
            try:
                parsed = self.gpt.structured_prompt(  # type: ignore[attr-defined]
                    prompt=prompt,
                    response_model=_AnswerSchema,  # type: ignore[arg-type]
                    model=model,
                    instructions=instructions,
                    reasoning_effort="low",
                )
                result_dict = parsed.model_dump()  # type: ignore[attr-defined]
                # Post-process: ensure only allowed ids
                if (
                    result_dict.get("room_id")
                    and result_dict["room_id"] not in candidate_room_ids
                ):
                    result_dict["room_id"] = None
                if (
                    result_dict.get("object_id")
                    and result_dict["object_id"] not in candidate_object_ids
                ):
                    result_dict["object_id"] = None
                result_dict["rooms"] = [
                    r for r in result_dict.get("rooms", []) if r in candidate_room_ids
                ]
                result_dict["objects"] = [
                    o
                    for o in result_dict.get("objects", [])
                    if o in candidate_object_ids
                ]
                if intent and not result_dict.get("intent"):
                    result_dict["intent"] = intent
                if return_text_only:
                    return result_dict.get("answer", "")
                return result_dict
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Structured answer generation failed (fallback to text): {}", e
                )

        # Fallback plain text prompt ---------------------------------------
        fallback_prompt = (
            "Provide a concise grounded answer. Then (if possible) name primary room id and object id explicitly.\n\n"
            + prompt
        )
        try:
            answer_text = self.gpt.text_prompt(  # type: ignore[attr-defined]
                fallback_prompt,
                model=model,
                instructions=instructions,
            )
        except Exception as e:  # pragma: no cover
            logger.error("Failed to generate (fallback) answer: {}", e)
            raise

        # Heuristic id extraction
        lower_ans = answer_text.lower()
        heuristic_room = None
        for rid in candidate_room_ids:
            if rid and rid.lower() in lower_ans:
                heuristic_room = rid
                break
        heuristic_object = None
        for oid in candidate_object_ids:
            if oid and oid.lower() in lower_ans:
                heuristic_object = oid
                break

        result_fallback = {
            "answer": answer_text,
            "intent": intent,
            "room_id": heuristic_room or (primary_room_id if not is_multi else None),
            "object_id": heuristic_object,
            "rooms": (
                candidate_room_ids
                if is_multi
                else ([primary_room_id] if primary_room_id else [])
            ),
            "objects": (
                candidate_object_ids
                if is_multi
                else ([heuristic_object] if heuristic_object else [])
            ),
            "_mode": "fallback",
        }
        if return_text_only:
            return answer_text
        return result_fallback

    # ------------------------------------------------------------------
    # Simple hierarchical lookup: floor -> room -> object with score fusion
    # ------------------------------------------------------------------
    def find_object_in_room_floor(
        self,
        query: str,
        *,
        top_k_floor: int = 3,
        top_k_room: int = 6,
        top_k_object: int = 40,
        require_floor_match: bool = True,
    ) -> Dict[str, Any]:
        """Given a query of the form 'the <object> in <room name> in floor <n>' perform:

        1. Regex parse (object, room, floor_number).
        2. Floor search (doc_type='floor'). Optionally restrict to parsed floor number.
        3. Room search within the chosen floor (doc_type='room'). Attempt fuzzy match on room name tokens.
        4. Object search (doc_type='object', object_modality='both'). Filter to chosen room.
        5. Combine text & object_visual scores using z-score normalization (when both present) for objects in that room.
        6. Return top combined object plus ranking diagnostics.

        Returns a dict with keys:
            parsed: {object, room, floor}
            floor: {id, score, metadata} (may be None)
            room:  {id, score, metadata} (may be None)
            top_object: {chunk_id, combined_score, text_score, visual_score, metadata} (may be None)
            rankings: list of the above per candidate object
            errors: list of warning strings
        """

        import re

        errors: List[str] = []

        pattern = re.compile(
            r"\bthe\s+(?P<object>.+?)\s+in\s+(?P<room>.+?)\s+in\s+floor\s+(?P<floor>\d+)\b",
            re.IGNORECASE,
        )
        m = pattern.search(query.strip())
        object_phrase = room_phrase = floor_num = None
        if m:
            object_phrase = m.group("object").strip()
            room_phrase = m.group("room").strip()
            floor_num = m.group("floor").strip()
        else:
            # Fallback heuristic split
            parts = [p.strip() for p in query.split(" in ")]
            if len(parts) >= 3 and parts[-1].lower().startswith("floor"):
                floor_num = re.sub(r"[^0-9]", "", parts[-1]) or None
                room_phrase = parts[-2]
                object_phrase = " ".join(parts[:-2]).replace("the", "", 1).strip()
            else:
                errors.append("Failed to parse query with expected pattern.")

        parsed = {
            "object": object_phrase,
            "room": room_phrase,
            "floor": floor_num,
        }

        def _z(arr: List[float]) -> List[float]:
            if not arr:
                return []
            a = np.asarray(arr, dtype="float32")
            mu = a.mean()
            sigma = a.std()
            if sigma <= 1e-9:
                return (a - mu).tolist()
            return ((a - mu) / sigma).tolist()

        # --- Floor search ---
        floor_hit = None
        floor_results = self.search(
            f"floor {floor_num}" if floor_num else (room_phrase or query),
            top_k=top_k_floor,
            doc_types=["floor"],
            frame_modality="text",
            object_modality="text",
        ).get("text", [])

        if floor_results:
            if floor_num and require_floor_match:
                # try to find floor whose metadata floor_id matches number
                matching = [
                    fr
                    for fr in floor_results
                    if str(
                        fr.chunk.metadata.get("floor_id")
                        or fr.chunk.metadata.get("floor")
                        or fr.chunk.id
                    )
                    == str(floor_num)
                ]
                floor_hit = matching[0] if matching else floor_results[0]
                if not matching:
                    errors.append(
                        "No exact floor_id match; using top floor search result instead."
                    )
            else:
                floor_hit = floor_results[0]
        else:
            errors.append("No floor results found.")

        floor_id = None
        if floor_hit:
            floor_id = (
                floor_hit.chunk.metadata.get("floor_id")
                or floor_hit.chunk.metadata.get("floor")
                or floor_hit.chunk.id
            )

        # --- Room search (restricted to floor) ---
        room_hit = None
        if floor_id:
            room_query = (
                f"{room_phrase} in floor {floor_num}"
                if room_phrase and floor_num
                else (room_phrase or query)
            )
            room_results = self.search(
                room_query,
                top_k=top_k_room,
                doc_types=["room"],
                frame_modality="text",
                object_modality="text",
            ).get("text", [])
            # Filter to matching floor
            floor_filtered = [
                rr
                for rr in room_results
                if (
                    rr.chunk.metadata.get("floor_id")
                    or rr.chunk.metadata.get("floor")
                    or rr.chunk.metadata.get("floorId")
                )
                == floor_id
            ]
            candidates = floor_filtered or room_results
            if candidates:
                if room_phrase:
                    # fuzzy token match heuristic
                    tokens = {t.lower() for t in re.findall(r"\w+", room_phrase)}
                    scored = []
                    for rr in candidates:
                        room_text = (
                            rr.chunk.metadata.get("room_name") or rr.chunk.content or ""
                        ).lower()
                        overlap = sum(1 for t in tokens if t in room_text)
                        scored.append((overlap, rr))
                    scored.sort(key=lambda x: (x[0], x[1].score), reverse=True)
                    room_hit = scored[0][1]
                else:
                    room_hit = candidates[0]
            else:
                errors.append("No room results found in selected floor.")
        else:
            errors.append("Skipping room search (no floor_id).")

        room_id = None
        if room_hit:
            room_id = (
                room_hit.chunk.metadata.get("room_id")
                or room_hit.chunk.metadata.get("room")
                or room_hit.chunk.id
            )

        # --- Object search (combine text + visual) ---
        rankings: List[Dict[str, Any]] = []
        top_object = None
        if room_id and object_phrase:
            object_query = (
                f"{object_phrase} in {room_phrase} in floor {floor_num}"
                if room_phrase and floor_num
                else object_phrase
            )
            obj_results = self.search(
                object_query,
                top_k=top_k_object,
                doc_types=["object"],
                object_modality="both",
                frame_modality="text",
            )
            text_objs = [
                sr
                for sr in obj_results.get("text", [])
                if sr.chunk.metadata.get("room_id") == room_id
            ]
            vis_objs = [
                sr
                for sr in obj_results.get("object_visual", [])
                if sr.chunk.metadata.get("room_id") == room_id
            ]
            # Index by chunk id
            by_id = {}
            for sr in text_objs:
                by_id.setdefault(sr.chunk.id, {})["text_score"] = sr.score
                by_id[sr.chunk.id]["chunk"] = sr.chunk
            for sr in vis_objs:
                by_id.setdefault(sr.chunk.id, {})["visual_score"] = sr.score
                by_id[sr.chunk.id]["chunk"] = sr.chunk
            if by_id:
                text_scores = [v.get("text_score", 0.0) for v in by_id.values()]
                vis_scores = [v.get("visual_score", 0.0) for v in by_id.values()]
                z_text = _z(text_scores)
                z_vis = _z(vis_scores)
                for (cid, v), zt, zv in zip(by_id.items(), z_text, z_vis):
                    # Normalize by modality count so multi-modal objects
                    # don't get an unfair additive boost over text-only ones.
                    if "visual_score" in v:
                        combined = (zt + zv) / 2.0
                    else:
                        combined = zt
                    rankings.append(
                        {
                            "chunk_id": cid,
                            "combined_score": float(combined),
                            "text_score": float(v.get("text_score", 0.0)),
                            "visual_score": float(v.get("visual_score", 0.0)),
                            "metadata": v["chunk"].metadata,
                        }
                    )
                rankings.sort(key=lambda x: x["combined_score"], reverse=True)
                if rankings:
                    top_object = rankings[0]
            else:
                errors.append("No object candidates found in selected room.")
        else:
            errors.append("Skipping object search (missing room_id or object phrase).")

        return {
            "floor_id": floor_id,
            "room_id": room_id,
            "object_id": top_object["chunk_id"] if top_object else None,
            "metadata": top_object["metadata"] if top_object else None,
        }
