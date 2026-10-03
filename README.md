> An earlier project. For current work, start with [DocExtract](https://github.com/ChunkyTortoise/docextract) and [ai-redteam-notes](https://github.com/ChunkyTortoise/ai-redteam-notes).

[![CI](https://github.com/ChunkyTortoise/graphrag-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/ChunkyTortoise/graphrag-demo/actions)

# GraphRAG Demo: Entity-Aware Multi-Hop Retrieval

RAG with knowledge-graph extraction for multi-hop questions, entity-aware retrieval, and confidence scoring.

![Knowledge Graph tab after indexing a sample doc (14 entities, BM25/no-API mode)](docs/screenshots/hero.png)

## What it shows

- **Retrieval beyond vector search:** knowledge-graph extraction (spaCy, regex or Claude), multi-hop retrieval over a NetworkX graph, and BM25 + graph fusion with reciprocal rank fusion.
- **Side-by-side comparison:** basic RAG vs. GraphRAG on the same question, in the Streamlit UI.
- **Runs without an API key:** BM25 mode needs no key.
- **Tested in CI:** 63 tests, with an 80% coverage floor ([`ci.yml`](.github/workflows/ci.yml)).

---

## Try It Now

Streamlit Cloud URL exists but is currently auth-gated (not a public try-it link). Run locally instead (BM25 mode needs no API key):

```bash
git clone https://github.com/ChunkyTortoise/graphrag-demo.git
cd graphrag-demo
pip install -e "."
streamlit run app.py
```

> To enable Claude-powered entity extraction: add your `ANTHROPIC_API_KEY` in the app sidebar.
> BM25 retrieval works without any API key using fast keyword-based graph construction.

### What to Try

1. Upload any `.txt` or `.pdf` document (try a Wikipedia article or a news story)
2. Ask a multi-hop question: *"How are [Entity A] and [Entity B] related?"*
3. Toggle between BM25-only and Claude extraction in the sidebar
4. Inspect the knowledge graph visualization - nodes are entities, edges are relationships

## What Makes It Different From Basic RAG

- **Entity Graph** - Extracts named entities (people, orgs, locations, products, concepts) and builds a NetworkX knowledge graph with co-occurrence relationships
- **Multi-Hop Retrieval** - Traverses the entity graph to find related chunks beyond keyword matching, enabling answers that connect information across document sections
- **Confidence Scores** - Quantifies answer reliability based on entity coverage and retrieval quality
- **Fact Checking** - Validates answer claims against source text with word-level overlap analysis
- **Side-by-Side Comparison** - Compare Basic RAG (BM25-only) vs GraphRAG on the same query

## Architecture

```mermaid
graph LR
    A[Document Upload] --> B[Text Chunking]
    B --> C{Entity Extraction}
    C --> C1[spaCy NER]
    C --> C2[Regex Patterns]
    C --> C3[Claude LLM]
    C1 & C2 & C3 --> D[Knowledge Graph<br/>NetworkX]
    B --> E[BM25 Index]
    D --> F[Graph Traversal<br/>Multi-hop]
    E --> G[Keyword Retrieval]
    F & G --> H[Score Fusion<br/>RRF]
    H --> I[Answer + Sources]
    I --> J[Fact Check<br/>Claude]

    style A fill:#4A90D9,color:#fff
    style D fill:#7B68EE,color:#fff
    style H fill:#50C878,color:#fff
    style J fill:#FF8C42,color:#fff
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

**Cayman Roden** - AI Engineer
[Portfolio](https://chunkytortoise.github.io) | [LinkedIn](https://www.linkedin.com/in/caymanroden/) | [GitHub](https://github.com/ChunkyTortoise)

## License

[MIT](LICENSE)
