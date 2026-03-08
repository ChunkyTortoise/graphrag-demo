![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![Claude](https://img.shields.io/badge/Claude-Sonnet-blueviolet)
![CI](https://github.com/ChunkyTortoise/graphrag-demo/actions/workflows/ci.yml/badge.svg)

# GraphRAG Demo: Entity-Aware Multi-Hop Retrieval

A production-grade RAG pipeline enhanced with knowledge graph extraction for multi-hop reasoning, entity-aware retrieval, and confidence scoring.

## What Makes It Different From Basic RAG

- **Entity Graph** — Extracts named entities (people, orgs, locations, products, concepts) and builds a NetworkX knowledge graph with co-occurrence relationships
- **Multi-Hop Retrieval** — Traverses the entity graph to find related chunks beyond keyword matching, enabling answers that connect information across document sections
- **Confidence Scores** — Quantifies answer reliability based on entity coverage and retrieval quality
- **Fact Checking** — Validates answer claims against source text with word-level overlap analysis
- **Side-by-Side Comparison** — Compare Basic RAG (BM25-only) vs GraphRAG on the same query

## Architecture

```
Document Upload (PDF / TXT / MD)
        |
        v
  Text Chunking (800 tokens, 100 overlap, sentence boundaries)
        |
        +---> Entity Extraction (spaCy NER / regex / Claude Haiku)
        |           |
        |           v
        |     Knowledge Graph (NetworkX DiGraph)
        |       - Nodes: entities with type + mention count
        |       - Edges: co-occurrence relationships
        |           |
        v           v
   BM25 Index   Graph Traversal (configurable hops)
        |           |
        +-----+-----+
              |
              v
    Weighted Score Fusion (BM25 + graph overlap)
              |
              v
    Answer + Confidence % + Sources + Graph Path
              |
              v
    Fact Check (claim vs source coverage)
```

## Run Locally

```bash
# Clone
git clone https://github.com/ChunkyTortoise/graphrag-demo.git
cd graphrag-demo

# Install
pip install -e ".[dev]"

# Optional: better entity extraction
pip install -e ".[nlp]"
python -m spacy download en_core_web_sm

# Optional: Claude-powered extraction + answer generation
export ANTHROPIC_API_KEY=sk-ant-...

# Run tests
pytest tests/ -x -q

# Launch
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Optionally add `ANTHROPIC_API_KEY` in Secrets for LLM-powered features
5. Deploy

## Project Structure

```
graphrag-demo/
├── graph/
│   ├── extractor.py       # Entity + relationship extraction
│   ├── knowledge_graph.py # NetworkX graph builder + serialization
│   └── retriever.py       # Graph-aware multi-hop retrieval
├── rag/
│   ├── basic.py           # Standard BM25 RAG pipeline
│   └── graph_rag.py       # GraphRAG pipeline with fact checking
├── tests/
│   ├── test_extractor.py
│   ├── test_knowledge_graph.py
│   └── test_retriever.py
├── app.py                 # Streamlit app (3 tabs: Chat, Graph, Comparison)
├── pyproject.toml
└── requirements.txt
```

## Built By

**Cayman Roden** — AI/ML Engineer
[LinkedIn](https://www.linkedin.com/in/caymanroden/) | [GitHub](https://github.com/ChunkyTortoise) | [Fiverr](https://www.fiverr.com/caymanroden)
