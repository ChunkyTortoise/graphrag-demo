"""In-memory vector store using sentence-transformers for cosine similarity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStore:
    """In-memory vector store with sentence-transformers embeddings.

    Uses 'all-MiniLM-L6-v2' (384-dim, ~80MB) for fast, quality embeddings.
    Falls back to deterministic random embeddings if sentence-transformers
    is not installed.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._documents: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info("Loaded embedding model: %s", self.MODEL_NAME)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers. "
                "Using random embeddings as fallback."
            )
            self._model = None

    def _embed(self, text: str) -> np.ndarray:
        if self._model is not None:
            return self._model.encode(text, show_progress_bar=False)
        # Fallback: deterministic random embedding based on text hash
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(384).astype(np.float32)

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the vector store."""
        if metadata is None:
            metadata = [{} for _ in documents]

        for doc, meta in zip(documents, metadata):
            embedding = self._embed(doc)
            self._documents.append(doc)
            self._embeddings.append(embedding)
            self._metadata.append(meta)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for top-k most similar documents."""
        if not self._documents:
            return []

        query_embedding = self._embed(query)
        scores = [
            _cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self._embeddings
        ]

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            {
                "document": self._documents[i],
                "score": scores[i],
                "metadata": self._metadata[i],
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
        self._embeddings.clear()
        self._metadata.clear()

    @property
    def count(self) -> int:
        return len(self._documents)
