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
from rag.vector_store import VectorStore


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
        self.vector_store = VectorStore()

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._kg

    @property
    def retriever(self) -> GraphRetriever:
        return self._retriever

    def ingest(self, doc_id: str, text: str) -> int:
        """Ingest a document: chunk, extract entities, build graph + BM25 + vector index."""
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        self._documents[doc_id] = text
        self._kg.add_document(doc_id, text, chunks)
        self._retriever.build_index()
        self.vector_store.add_documents(
            chunks, [{"doc_id": doc_id, "chunk_idx": i} for i in range(len(chunks))]
        )
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

    @staticmethod
    def _rrf_score(bm25_rank: int, vector_rank: int, k: int = 60) -> float:
        """Reciprocal Rank Fusion score."""
        return 1.0 / (k + bm25_rank) + 1.0 / (k + vector_rank)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid retrieval: BM25 + vector similarity via RRF."""
        # BM25 results from the graph retriever (already ranked)
        bm25_results = self._retriever.retrieve(query, k=top_k * 2)
        bm25_ranks: dict[str, int] = {
            r.text: i + 1 for i, r in enumerate(bm25_results)
        }

        # Vector results
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        vector_ranks: dict[str, int] = {
            r["document"]: r["rank"] for r in vector_results
        }

        # Combine via RRF
        all_docs = set(list(bm25_ranks.keys()) + list(vector_ranks.keys()))
        fallback_rank = top_k * 2 + 1
        scored: list[dict] = []
        for doc in all_docs:
            bm25_r = bm25_ranks.get(doc, fallback_rank)
            vec_r = vector_ranks.get(doc, fallback_rank)
            rrf = self._rrf_score(bm25_r, vec_r)
            scored.append({
                "document": doc,
                "rrf_score": rrf,
                "bm25_rank": bm25_r,
                "vector_rank": vec_r,
            })

        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored[:top_k]

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
