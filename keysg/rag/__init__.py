"""
RAG (Retrieval-Augmented Generation) module for HovFun.

This module provides:
- GraphContextRetriever: Hierarchical text-based retrieval using embeddings
- VisualContextRetriever: Visual context retrieval using CLIP embeddings
- Query analysis utilities for understanding natural language queries

These components implement Section III-E of the KeySG paper: Scene Querying
and Hierarchical RAG.
"""

from .visual_context_retriever import VisualContextRetriever, FrameData, RetrievalResult
from .graph_context_retriever import GraphContextRetriever, SearchResult

__version__ = "1.0.0"

__all__ = [
    "VisualContextRetriever",
    "FrameData",
    "RetrievalResult",
    "GraphContextRetriever",
    "SearchResult",
]
