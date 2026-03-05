"""Tests for entity and relationship extraction."""

import pytest

from graph.extractor import Entity, EntityExtractor, EntityType, Relationship


SAMPLE_TEXT = (
    "John Smith works at Google in Mountain View. "
    "Google released the Gemini Model last year. "
    "Sarah Johnson from Microsoft visited the Google campus. "
    "The Gemini Model competes with OpenAI products."
)

SAMPLE_TEXT_ORGS = (
    "Apple Inc announced a partnership with Tesla Corp. "
    "Microsoft Corp and Amazon Inc are also competitors. "
    "Google LLC dominates the search market."
)


@pytest.fixture
def extractor() -> EntityExtractor:
    return EntityExtractor(use_llm=False)


class TestEntityExtraction:
    def test_extract_returns_entities(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        assert len(entities) > 0
        assert all(isinstance(e, Entity) for e in entities)

    def test_entity_has_required_fields(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        for entity in entities:
            assert entity.name
            assert isinstance(entity.entity_type, EntityType)
            assert entity.mentions >= 1

    def test_extracts_person_names(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        names = [e.name for e in entities if e.entity_type == EntityType.PERSON]
        assert any("John" in n for n in names) or any("Smith" in n for n in names)

    def test_extracts_organizations(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT_ORGS)
        org_names = [e.name.lower() for e in entities if e.entity_type == EntityType.ORG]
        # At least one org should be found
        assert len(org_names) > 0

    def test_entity_id_format(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        for entity in entities:
            assert ":" in entity.id
            parts = entity.id.split(":", 1)
            assert parts[0] in [t.value for t in EntityType]

    def test_extract_with_chunk_id(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT, chunk_id=5)
        for entity in entities:
            assert 5 in entity.source_chunks

    def test_empty_text_returns_empty(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities("")
        assert entities == []

    def test_short_text(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities("hello world")
        assert isinstance(entities, list)


class TestRelationshipExtraction:
    def test_extract_relationships(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        rels = extractor.extract_relationships(SAMPLE_TEXT, entities)
        assert isinstance(rels, list)

    def test_relationship_has_required_fields(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        rels = extractor.extract_relationships(SAMPLE_TEXT, entities)
        for rel in rels:
            assert isinstance(rel, Relationship)
            assert isinstance(rel.source, Entity)
            assert isinstance(rel.target, Entity)
            assert rel.relation_type
            assert rel.weight > 0

    def test_co_occurrence_relationships(self, extractor: EntityExtractor) -> None:
        text = "Alice Brown and Bob White met at Google Inc campus."
        entities = extractor.extract_entities(text)
        if len(entities) >= 2:
            rels = extractor.extract_relationships(text, entities)
            assert len(rels) > 0
            assert all(r.relation_type == "co-occurs" for r in rels)

    def test_no_duplicate_relationships(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract_entities(SAMPLE_TEXT)
        rels = extractor.extract_relationships(SAMPLE_TEXT, entities)
        pairs = [(r.source.id, r.target.id) for r in rels]
        assert len(pairs) == len(set(pairs))

    def test_empty_entities_returns_empty(self, extractor: EntityExtractor) -> None:
        rels = extractor.extract_relationships(SAMPLE_TEXT, [])
        assert rels == []
