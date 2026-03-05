"""GraphRAG pipeline integrating knowledge graph retrieval."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from graph.extractor import EntityExtractor
from graph.knowledge_graph import KnowledgeGraph
from graph.retriever import GraphRetriever, RetrievedChunk
from rag.basic import chunk_text


@dataclass
class GraphRAGResult:
    answer: str
    entities_found: list[str]
    graph_path: list[str]
    confidence: float
    sources: list[RetrievedChunk]
    latency_ms: int


@dataclass
class FactCheckResult:
    is_supported: bool
    support_score: float
    supporting_sources: list[str]
    unsupported_claims: list[str]


class GraphRAGPipeline:
    """RAG pipeline enhanced with knowledge graph for multi-hop reasoning."""

    def __init__(
        self,
        use_llm: bool | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self._use_llm = use_llm if use_llm is not None else bool(
            os.environ.get("ANTHROPIC_API_KEY")
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._extractor = EntityExtractor(use_llm=False)  # extraction always local
        self._kg = KnowledgeGraph(extractor=self._extractor)
        self._retriever = GraphRetriever(self._kg)
        self._documents: dict[str, str] = {}

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._kg

    @property
    def retriever(self) -> GraphRetriever:
        return self._retriever

    def ingest(self, doc_id: str, text: str) -> int:
        """Ingest a document: chunk, extract entities, build graph + BM25 index."""
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        self._documents[doc_id] = text
        self._kg.add_document(doc_id, text, chunks)
        self._retriever.build_index()
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> GraphRAGResult:
        """Query the pipeline: retrieve via graph + BM25, generate answer."""
        start = time.time()

        # Retrieve
        chunks = self._retriever.retrieve(question, k=top_k)

        # Find entities in query
        query_entity_ids = self._kg.find_entities_in_query(question)
        entity_names = []
        for eid in query_entity_ids:
            if eid in self._kg.graph.nodes:
                entity_names.append(self._kg.graph.nodes[eid].get("name", eid))

        # Build graph traversal path
        graph_path = self._build_graph_path(query_entity_ids)

        # Confidence
        confidence = self._retriever.confidence_score(question, chunks)

        # Generate answer
        if self._use_llm and chunks:
            answer = self._generate_with_llm(question, chunks)
        elif chunks:
            answer = self._generate_extractive(question, chunks)
        else:
            answer = "No relevant information found in the indexed documents."

        latency_ms = int((time.time() - start) * 1000)

        return GraphRAGResult(
            answer=answer,
            entities_found=entity_names,
            graph_path=graph_path,
            confidence=confidence,
            sources=chunks,
            latency_ms=latency_ms,
        )

    def fact_check(self, answer: str, sources: list[RetrievedChunk]) -> FactCheckResult:
        """Compare answer claims against source text."""
        if not sources:
            return FactCheckResult(
                is_supported=False,
                support_score=0.0,
                supporting_sources=[],
                unsupported_claims=[answer],
            )

        # Split answer into sentences as "claims"
        claims = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip() and len(s.strip()) > 10]
        if not claims:
            return FactCheckResult(
                is_supported=True,
                support_score=1.0,
                supporting_sources=[],
                unsupported_claims=[],
            )

        source_texts = [s.text.lower() for s in sources]
        combined_source = " ".join(source_texts)

        supported_claims: list[str] = []
        unsupported_claims: list[str] = []
        supporting_source_ids: set[str] = set()

        for claim in claims:
            # Check if key words from claim appear in sources
            claim_words = set(claim.lower().split())
            # Remove stopwords
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                         "to", "for", "of", "and", "or", "but", "with", "this", "that",
                         "it", "its", "by", "from", "as", "be", "been", "has", "have",
                         "had", "not", "no", "can", "will", "would", "could", "should"}
            content_words = claim_words - stopwords
            if not content_words:
                supported_claims.append(claim)
                continue

            overlap = sum(1 for w in content_words if w in combined_source)
            coverage = overlap / len(content_words) if content_words else 0

            if coverage >= 0.5:
                supported_claims.append(claim)
                for i, src_text in enumerate(source_texts):
                    if any(w in src_text for w in content_words):
                        supporting_source_ids.add(f"chunk_{sources[i].chunk_id}")
            else:
                unsupported_claims.append(claim)

        total = len(supported_claims) + len(unsupported_claims)
        support_score = len(supported_claims) / total if total > 0 else 0.0

        return FactCheckResult(
            is_supported=support_score >= 0.5,
            support_score=round(support_score, 3),
            supporting_sources=sorted(supporting_source_ids),
            unsupported_claims=unsupported_claims,
        )

    def _build_graph_path(self, entity_ids: list[str]) -> list[str]:
        """Build a traversal path description for display."""
        path: list[str] = []
        for eid in entity_ids:
            if eid not in self._kg.graph.nodes:
                continue
            node_name = self._kg.graph.nodes[eid].get("name", eid)
            neighbors = self._kg.get_entity_neighbors(eid, hops=1)
            neighbor_names = []
            for nid in neighbors[:3]:
                if nid in self._kg.graph.nodes:
                    neighbor_names.append(self._kg.graph.nodes[nid].get("name", nid))
            if neighbor_names:
                path.append(f"{node_name} -> [{', '.join(neighbor_names)}]")
            else:
                path.append(node_name)
        return path

    def _generate_extractive(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an extractive answer from top chunks."""
        if not chunks:
            return "No relevant information found."

        # Use top chunk, trimmed
        top = chunks[0]
        answer = top.text[:600]

        # Add source references
        source_refs = ", ".join(f"[Source {i+1}]" for i in range(min(len(chunks), 3)))
        return f"{answer}\n\n(Based on {source_refs})"

    def _generate_with_llm(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate answer using Claude."""
        import anthropic

        sources_text = "\n\n".join(
            f"[Source {i + 1}]\n{chunk.text[:500]}" for i, chunk in enumerate(chunks)
        )

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based on the provided sources. "
            "Reference sources as [Source 1], [Source 2], etc. Be concise and accurate. "
            "If the answer is not in the sources, say so clearly."
        )
        user_message = f"Sources:\n{sources_text}\n\nQuestion: {question}"

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception:
            return self._generate_extractive(question, chunks)
