"""Standard BM25 + embedding RAG pipeline (adapted from source)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi


@dataclass
class BasicSourceChunk:
    content: str
    score: float
    chunk_index: int


@dataclass
class BasicRAGResponse:
    answer: str
    sources: list[BasicSourceChunk]
    latency_ms: int


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks using sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_len + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = (
                current_chunk[-chunk_overlap:]
                if len(current_chunk) > chunk_overlap
                else current_chunk[:]
            )
            current_chunk = overlap_words + words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text[:4000]]


class BasicRAGPipeline:
    """BM25-based RAG pipeline that works without external APIs."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._corpus: list[str] = []
        self._bm25: BM25Okapi | None = None

    @property
    def corpus(self) -> list[str]:
        return self._corpus

    def ingest(self, text: str) -> int:
        """Chunk and index text. Returns number of chunks."""
        self._corpus = chunk_text(text, self._chunk_size, self._chunk_overlap)
        tokenized = [c.lower().split() for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        return len(self._corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[BasicSourceChunk]:
        """Retrieve top-k chunks by BM25 score."""
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_q = query.lower().split()
        scores = self._bm25.get_scores(tokenized_q)
        max_score = max(scores) if any(s > 0 for s in scores) else 1.0

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            BasicSourceChunk(
                content=self._corpus[i][:200],
                score=round(float(scores[i] / (max_score + 1e-9)), 4),
                chunk_index=i,
            )
            for i in ranked
            if scores[i] > 0
        ]

    def query(self, question: str, top_k: int = 5) -> BasicRAGResponse:
        """Retrieve chunks and generate answer (extractive, no LLM)."""
        start = time.time()
        sources = self.retrieve(question, top_k)

        if not sources:
            answer = "No relevant information found in the document."
        else:
            # Extractive answer: return top chunk as the answer
            top_chunk_idx = sources[0].chunk_index
            answer = self._corpus[top_chunk_idx][:500]

        latency_ms = int((time.time() - start) * 1000)
        return BasicRAGResponse(answer=answer, sources=sources, latency_ms=latency_ms)
