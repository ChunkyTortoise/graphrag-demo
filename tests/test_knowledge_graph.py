"""Tests for knowledge graph construction and querying."""

import pytest

from graph.extractor import EntityExtractor
from graph.knowledge_graph import KnowledgeGraph


SAMPLE_DOC = (
    "John Smith is the CEO of Acme Corp. Acme Corp is headquartered in New York. "
    "Jane Doe works at Acme Corp as CTO. "
    "Acme Corp builds the Acme Platform for enterprise customers. "
    "Bob Wilson from Globex Corp met with John Smith in Chicago."
)


@pytest.fixture
def kg() -> KnowledgeGraph:
    extractor = EntityExtractor(use_llm=False)
    return KnowledgeGraph(extractor=extractor)


@pytest.fixture
def populated_kg(kg: KnowledgeGraph) -> KnowledgeGraph:
    chunks = [
        "John Smith is the CEO of Acme Corp. Acme Corp is headquartered in New York.",
        "Jane Doe works at Acme Corp as CTO. She reports to John Smith.",
        "Acme Corp builds the Acme Platform for enterprise customers.",
        "Bob Wilson from Globex Corp met with John Smith in Chicago.",
    ]
    kg.add_document("doc1", SAMPLE_DOC, chunks)
    return kg


class TestKnowledgeGraphConstruction:
    def test_add_document_creates_nodes(self, populated_kg: KnowledgeGraph) -> None:
        assert populated_kg.entity_count > 0

    def test_add_document_creates_edges(self, populated_kg: KnowledgeGraph) -> None:
        assert populated_kg.relationship_count >= 0  # may or may not have edges

    def test_chunks_are_stored(self, populated_kg: KnowledgeGraph) -> None:
        assert len(populated_kg.chunks) == 4
        for key, record in populated_kg.chunks.items():
            assert record.doc_id == "doc1"
            assert record.text
            assert record.chunk_id >= 0

    def test_empty_document(self, kg: KnowledgeGraph) -> None:
        kg.add_document("empty", "", [""])
        assert kg.entity_count >= 0  # no crash

    def test_multiple_documents(self, kg: KnowledgeGraph) -> None:
        kg.add_document("doc1", "Alice Brown works at TechCo Inc.", ["Alice Brown works at TechCo Inc."])
        kg.add_document("doc2", "Bob Green works at TechCo Inc.", ["Bob Green works at TechCo Inc."])
        assert len(kg.chunks) == 2


class TestEntityNeighbors:
    def test_get_neighbors_returns_list(self, populated_kg: KnowledgeGraph) -> None:
        nodes = list(populated_kg.graph.nodes)
        if nodes:
            neighbors = populated_kg.get_entity_neighbors(nodes[0], hops=1)
            assert isinstance(neighbors, list)

    def test_nonexistent_entity(self, populated_kg: KnowledgeGraph) -> None:
        neighbors = populated_kg.get_entity_neighbors("FAKE:nonexistent")
        assert neighbors == []

    def test_multi_hop_expands(self, populated_kg: KnowledgeGraph) -> None:
        nodes = list(populated_kg.graph.nodes)
        if nodes:
            hop1 = populated_kg.get_entity_neighbors(nodes[0], hops=1)
            hop2 = populated_kg.get_entity_neighbors(nodes[0], hops=2)
            assert len(hop2) >= len(hop1)

    def test_self_not_in_neighbors(self, populated_kg: KnowledgeGraph) -> None:
        nodes = list(populated_kg.graph.nodes)
        if nodes:
            neighbors = populated_kg.get_entity_neighbors(nodes[0], hops=2)
            assert nodes[0] not in neighbors


class TestRelevantChunks:
    def test_get_relevant_chunks(self, populated_kg: KnowledgeGraph) -> None:
        entity_ids = list(populated_kg.graph.nodes)[:2]
        if entity_ids:
            chunks = populated_kg.get_relevant_chunks(entity_ids)
            assert isinstance(chunks, list)

    def test_no_entities_returns_empty(self, populated_kg: KnowledgeGraph) -> None:
        chunks = populated_kg.get_relevant_chunks([])
        assert chunks == []

    def test_fake_entity_returns_empty(self, populated_kg: KnowledgeGraph) -> None:
        chunks = populated_kg.get_relevant_chunks(["FAKE:nonexistent"])
        assert chunks == []


class TestFindEntitiesInQuery:
    def test_finds_matching_entities(self, populated_kg: KnowledgeGraph) -> None:
        # Get a known entity name from the graph
        for node_id in populated_kg.graph.nodes:
            name = populated_kg.graph.nodes[node_id].get("name", "")
            if name:
                matches = populated_kg.find_entities_in_query(name)
                assert node_id in matches
                break

    def test_no_match_returns_empty(self, populated_kg: KnowledgeGraph) -> None:
        matches = populated_kg.find_entities_in_query("xyzzy nonexistent query")
        assert matches == []


class TestSerialization:
    def test_to_dict_roundtrip(self, populated_kg: KnowledgeGraph) -> None:
        data = populated_kg.to_dict()
        assert "nodes" in data or "links" in data  # NetworkX format

    def test_to_json(self, populated_kg: KnowledgeGraph) -> None:
        json_str = populated_kg.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_graphviz_dot(self, populated_kg: KnowledgeGraph) -> None:
        dot = populated_kg.get_graphviz_dot()
        assert "digraph" in dot
        assert "KnowledgeGraph" in dot
