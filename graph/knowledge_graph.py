"""Knowledge graph construction and querying with NetworkX."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import networkx as nx

from graph.extractor import Entity, EntityExtractor, Relationship


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: int
    text: str
    entities: list[Entity] = field(default_factory=list)


class KnowledgeGraph:
    """Builds and queries a knowledge graph from documents.

    Nodes are entities; edges are relationships.
    Each node stores which chunks it appears in.
    """

    def __init__(self, extractor: EntityExtractor | None = None) -> None:
        self._graph = nx.DiGraph()
        self._extractor = extractor or EntityExtractor(use_llm=False)
        self._chunks: dict[str, ChunkRecord] = {}  # key: "doc_id:chunk_id"

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def chunks(self) -> dict[str, ChunkRecord]:
        return self._chunks

    @property
    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relationship_count(self) -> int:
        return self._graph.number_of_edges()

    def add_document(self, doc_id: str, text: str, chunks: list[str]) -> None:
        """Extract entities from each chunk and add to the graph."""
        for i, chunk_text in enumerate(chunks):
            chunk_key = f"{doc_id}:{i}"
            entities = self._extractor.extract_entities(chunk_text, chunk_id=i)
            relationships = self._extractor.extract_relationships(
                chunk_text, entities, chunk_id=i
            )

            self._chunks[chunk_key] = ChunkRecord(
                doc_id=doc_id, chunk_id=i, text=chunk_text, entities=entities
            )

            # Add entity nodes
            for entity in entities:
                if self._graph.has_node(entity.id):
                    node_data = self._graph.nodes[entity.id]
                    node_data["mentions"] = node_data.get("mentions", 0) + entity.mentions
                    existing_chunks = node_data.get("source_chunks", [])
                    for sc in entity.source_chunks:
                        chunk_ref = f"{doc_id}:{sc}"
                        if chunk_ref not in existing_chunks:
                            existing_chunks.append(chunk_ref)
                    node_data["source_chunks"] = existing_chunks
                else:
                    self._graph.add_node(
                        entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type.value,
                        mentions=entity.mentions,
                        source_chunks=[f"{doc_id}:{sc}" for sc in entity.source_chunks],
                    )

            # Add relationship edges
            for rel in relationships:
                if self._graph.has_edge(rel.source.id, rel.target.id):
                    edge_data = self._graph.edges[rel.source.id, rel.target.id]
                    edge_data["weight"] = edge_data.get("weight", 0) + rel.weight
                else:
                    self._graph.add_edge(
                        rel.source.id,
                        rel.target.id,
                        relation_type=rel.relation_type,
                        weight=rel.weight,
                        source_chunks=[f"{doc_id}:{sc}" for sc in rel.source_chunks],
                    )

    def get_entity_neighbors(self, entity_id: str, hops: int = 2) -> list[str]:
        """Multi-hop traversal: return entity IDs within N hops."""
        if entity_id not in self._graph:
            return []

        visited: set[str] = set()
        frontier = {entity_id}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                # Traverse both directions (successors + predecessors)
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited

        visited.update(frontier)
        visited.discard(entity_id)
        return list(visited)

    def get_relevant_chunks(self, query_entities: list[str]) -> list[ChunkRecord]:
        """Find chunks connected to query entities, ranked by entity overlap count."""
        chunk_scores: dict[str, int] = {}

        for entity_id in query_entities:
            if entity_id not in self._graph:
                continue
            node_data = self._graph.nodes[entity_id]
            for chunk_key in node_data.get("source_chunks", []):
                chunk_scores[chunk_key] = chunk_scores.get(chunk_key, 0) + 1

        # Sort by overlap count descending
        sorted_keys = sorted(chunk_scores, key=lambda k: chunk_scores[k], reverse=True)
        return [self._chunks[k] for k in sorted_keys if k in self._chunks]

    def find_entities_in_query(self, query: str) -> list[str]:
        """Find graph entity IDs that match terms in the query."""
        query_lower = query.lower()
        matched: list[str] = []
        for node_id in self._graph.nodes:
            node_data = self._graph.nodes[node_id]
            name = node_data.get("name", "").lower()
            if name and name in query_lower:
                matched.append(node_id)
        return matched

    def to_dict(self) -> dict:
        """Serialize graph to dict for JSON export."""
        return nx.node_link_data(self._graph)

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        """Deserialize graph from dict."""
        kg = cls()
        kg._graph = nx.node_link_graph(data)
        return kg

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def get_graphviz_dot(self) -> str:
        """Generate DOT string for Graphviz rendering."""
        type_colors = {
            "PERSON": "#4fc3f7",
            "ORG": "#81c784",
            "PRODUCT": "#ffb74d",
            "CONCEPT": "#ce93d8",
            "LOCATION": "#ef9a9a",
        }

        lines = ['digraph KnowledgeGraph {', '  rankdir=LR;', '  node [shape=box, style=filled];']

        for node_id in self._graph.nodes:
            data = self._graph.nodes[node_id]
            name = data.get("name", node_id)
            etype = data.get("entity_type", "CONCEPT")
            color = type_colors.get(etype, "#e0e0e0")
            safe_name = name.replace('"', '\\"')
            safe_id = node_id.replace('"', '\\"')
            lines.append(f'  "{safe_id}" [label="{safe_name}\\n({etype})", fillcolor="{color}"];')

        for src, tgt in self._graph.edges:
            edge_data = self._graph.edges[src, tgt]
            label = edge_data.get("relation_type", "")
            safe_src = src.replace('"', '\\"')
            safe_tgt = tgt.replace('"', '\\"')
            lines.append(f'  "{safe_src}" -> "{safe_tgt}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)
