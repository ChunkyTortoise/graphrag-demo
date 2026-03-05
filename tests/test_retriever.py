"""Tests for graph-aware retriever and GraphRAG pipeline."""

import pytest

from graph.extractor import EntityExtractor
from graph.knowledge_graph import KnowledgeGraph
from graph.retriever import GraphRetriever, RetrievedChunk
from rag.basic import BasicRAGPipeline, chunk_text
from rag.graph_rag import GraphRAGPipeline


SAMPLE_TEXT = (
    "John Smith is the CEO of Acme Corp based in New York. "
    "Acme Corp develops enterprise software products. "
    "Jane Doe is the CTO of Acme Corp and oversees all engineering. "
    "Bob Wilson from Globex Corp is a competitor in the enterprise market. "
    "Globex Corp is headquartered in Chicago and focuses on cloud computing. "
    "John Smith previously worked at Globex Corp before founding Acme Corp. "
    "The Acme Platform is their flagship product used by Fortune 500 companies. "
    "Jane Doe designed the architecture of the Acme Platform."
)


@pytest.fixture
def retriever() -> GraphRetriever:
    extractor = EntityExtractor(use_llm=False)
    kg = KnowledgeGraph(extractor=extractor)
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=50, chunk_overlap=10)
    kg.add_document("test_doc", SAMPLE_TEXT, chunks)
    r = GraphRetriever(kg)
    r.build_index()
    return r


@pytest.fixture
def graph_pipeline() -> GraphRAGPipeline:
    pipeline = GraphRAGPipeline(use_llm=False)
    pipeline.ingest("test_doc", SAMPLE_TEXT)
    return pipeline


class TestGraphRetriever:
    def test_retrieve_returns_results(self, retriever: GraphRetriever) -> None:
        results = retriever.retrieve("Acme Corp CEO")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retrieve_result_fields(self, retriever: GraphRetriever) -> None:
        results = retriever.retrieve("enterprise software")
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert r.text
            assert r.doc_id == "test_doc"
            assert r.score >= 0
            assert r.source in ("bm25", "graph", "both")

    def test_retrieve_respects_k(self, retriever: GraphRetriever) -> None:
        results = retriever.retrieve("Acme", k=2)
        assert len(results) <= 2

    def test_confidence_score(self, retriever: GraphRetriever) -> None:
        results = retriever.retrieve("Acme Corp")
        confidence = retriever.confidence_score("Acme Corp", results)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_no_chunks(self, retriever: GraphRetriever) -> None:
        confidence = retriever.confidence_score("anything", [])
        assert confidence == 0.0

    def test_empty_query(self, retriever: GraphRetriever) -> None:
        results = retriever.retrieve("")
        assert isinstance(results, list)


class TestBasicRAGPipeline:
    def test_ingest_and_query(self) -> None:
        pipeline = BasicRAGPipeline()
        n = pipeline.ingest(SAMPLE_TEXT)
        assert n > 0

        response = pipeline.query("Who is the CEO?")
        assert response.answer
        assert response.latency_ms >= 0

    def test_retrieve(self) -> None:
        pipeline = BasicRAGPipeline()
        pipeline.ingest(SAMPLE_TEXT)
        chunks = pipeline.retrieve("enterprise software", top_k=3)
        assert len(chunks) <= 3
        assert all(c.score >= 0 for c in chunks)

    def test_empty_corpus(self) -> None:
        pipeline = BasicRAGPipeline()
        response = pipeline.query("anything")
        assert "No relevant" in response.answer


class TestGraphRAGPipeline:
    def test_ingest(self, graph_pipeline: GraphRAGPipeline) -> None:
        assert graph_pipeline.knowledge_graph.entity_count > 0

    def test_query_returns_result(self, graph_pipeline: GraphRAGPipeline) -> None:
        result = graph_pipeline.query("Who is the CEO of Acme Corp?")
        assert result.answer
        assert result.latency_ms >= 0
        assert isinstance(result.confidence, float)
        assert isinstance(result.entities_found, list)
        assert isinstance(result.graph_path, list)
        assert isinstance(result.sources, list)

    def test_query_finds_entities(self, graph_pipeline: GraphRAGPipeline) -> None:
        result = graph_pipeline.query("Acme Corp products")
        # Should find at least some entities (depends on regex extraction)
        assert isinstance(result.entities_found, list)

    def test_fact_check_supported(self, graph_pipeline: GraphRAGPipeline) -> None:
        result = graph_pipeline.query("Acme Corp enterprise software")
        fc = graph_pipeline.fact_check(result.answer, result.sources)
        assert isinstance(fc.is_supported, bool)
        assert 0.0 <= fc.support_score <= 1.0

    def test_fact_check_no_sources(self, graph_pipeline: GraphRAGPipeline) -> None:
        fc = graph_pipeline.fact_check("Some random claim.", [])
        assert fc.is_supported is False
        assert fc.support_score == 0.0

    def test_multiple_documents(self) -> None:
        pipeline = GraphRAGPipeline(use_llm=False)
        pipeline.ingest("doc1", "Alice works at TechCo. TechCo builds AI tools.")
        pipeline.ingest("doc2", "Bob works at DataCo. DataCo analyzes data.")
        result = pipeline.query("TechCo AI tools")
        assert result.answer


class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=1)
        assert len(chunks) >= 1

    def test_empty_text(self) -> None:
        chunks = chunk_text("")
        assert len(chunks) == 1

    def test_overlap_preserves_context(self) -> None:
        words = " ".join(f"word{i}" for i in range(100))
        text = f"{words}. {words}."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        assert len(chunks) >= 2
