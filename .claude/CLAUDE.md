# GraphRAG Demo

## Stack
NetworkX | spaCy | Streamlit | Anthropic | BM25 | sentence-transformers (all-MiniLM-L6-v2) | pyvis | Python

## Architecture
Entity extraction (spaCy/regex/Claude) → NetworkX DiGraph → BM25+graph fusion retrieval (RRF) → fact-checking. 3-tab Streamlit UI: Chat / Graph / Comparison (Basic vs GraphRAG side-by-side). Interactive pyvis graph visualization.
- `app.py` — Streamlit entry point
- `graphrag/` — extraction, indexing, retrieval modules
- `vector_store/` — MiniLM embeddings

## Deploy
Streamlit Cloud — https://graphr-azucpb9e7jkdxavkahvicr.streamlit.app
Repo file: `app.py`. Needs `ANTHROPIC_API_KEY` in Streamlit secrets.

## Test
```pytest tests/  # 63 tests```

## Key Env
ANTHROPIC_API_KEY
