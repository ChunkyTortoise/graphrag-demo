"""
GraphRAG Demo — Entity-Aware Multi-Hop Retrieval
by Cayman Roden

Upload documents, build a knowledge graph, and compare Basic RAG vs GraphRAG.
"""

from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from graph.extractor import EntityExtractor, EntityType
from graph.knowledge_graph import KnowledgeGraph
from rag.basic import BasicRAGPipeline
from rag.graph_rag import GraphRAGPipeline


# -- Page config --
st.set_page_config(
    page_title="GraphRAG Demo — Cayman Roden",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Dark theme CSS --
st.markdown(
    """
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .main .block-container { background-color: #0f172a; padding-top: 2rem; }
    .stButton > button {
        background-color: #6366f1; color: white; border: none;
        border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stButton > button:hover { background-color: #4f46e5; }
    .metric-card { background-color: #1e293b; border-radius: 8px; padding: 1rem; text-align: center; }
    .success-box { background-color: #064e3b; border: 1px solid #10b981; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .info-box { background-color: #1e3a5f; border: 1px solid #6366f1; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    h1, h2, h3 { color: #f8fafc; }
</style>
""",
    unsafe_allow_html=True,
)


# -- Session state --
def _init_session() -> None:
    defaults: dict = {
        "graph_pipeline": None,
        "basic_pipeline": None,
        "doc_loaded": False,
        "filename": None,
        "messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# -- Sidebar --
with st.sidebar:
    st.markdown("## GraphRAG Demo")
    st.markdown("**by Cayman Roden**")
    st.markdown("*Entity-Aware Multi-Hop Retrieval*")
    st.divider()

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or Markdown files",
    )

    if st.button("Build Index", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Extracting entities and building knowledge graph..."):
                graph_pipeline = GraphRAGPipeline(use_llm=False)
                basic_pipeline = BasicRAGPipeline()
                all_text = ""

                for uf in uploaded_files:
                    content = uf.read()
                    ext = uf.name.lower().rsplit(".", 1)[-1] if "." in uf.name else "txt"

                    if ext == "pdf":
                        from pypdf import PdfReader
                        reader = PdfReader(io.BytesIO(content))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    else:
                        try:
                            text = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text = content.decode("latin-1")

                    text = text.strip()
                    if text:
                        graph_pipeline.ingest(uf.name, text)
                        all_text += f"\n\n{text}"

                if all_text.strip():
                    basic_pipeline.ingest(all_text.strip())

                st.session_state.graph_pipeline = graph_pipeline
                st.session_state.basic_pipeline = basic_pipeline
                st.session_state.doc_loaded = True
                st.session_state.filename = ", ".join(f.name for f in uploaded_files)
                st.session_state.messages = []

            st.success(f"Indexed {len(uploaded_files)} file(s)")

    if st.session_state.doc_loaded:
        st.divider()
        kg = st.session_state.graph_pipeline.knowledge_graph
        st.metric("Entities", kg.entity_count)
        st.metric("Relationships", kg.relationship_count)
        st.metric("Chunks", len(kg.chunks))


# -- Main content --
st.title("GraphRAG Demo — Entity-Aware Multi-Hop Retrieval")

if not st.session_state.doc_loaded:
    st.markdown(
        """
<div class="info-box">
Upload documents in the sidebar and click <strong>Build Index</strong> to get started.
<br><br>
GraphRAG enhances standard RAG by extracting entities and relationships into a knowledge graph,
enabling multi-hop reasoning and higher-confidence answers.
</div>
""",
        unsafe_allow_html=True,
    )

    # Architecture overview
    st.markdown("### How GraphRAG Works")
    st.code(
        """
Document Upload
      |
      v
Text Chunking (800 tokens, 100 overlap)
      |
      +---> Entity Extraction (spaCy / regex / Claude)
      |           |
      |           v
      |     Knowledge Graph (NetworkX DiGraph)
      |           |
      v           v
   BM25 Index   Graph Traversal (multi-hop)
      |           |
      +-----+-----+
            |
            v
    Combined Retrieval + Reranking
            |
            v
    Answer + Confidence + Sources + Graph Path
""",
        language="text",
    )
    st.stop()


# -- Tabs --
tab_chat, tab_graph, tab_compare = st.tabs(["Chat", "Knowledge Graph", "Comparison"])

# -- Tab 1: Chat --
with tab_chat:
    st.markdown(f"**Document**: `{st.session_state.filename}`")
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pipeline: GraphRAGPipeline = st.session_state.graph_pipeline
                result = pipeline.query(prompt)

                st.markdown(result.answer)

                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result.confidence * 100:.1f}%")
                col2.metric("Latency", f"{result.latency_ms}ms")
                col3.metric("Sources", len(result.sources))

                if result.entities_found:
                    st.markdown(f"**Entities found**: {', '.join(result.entities_found)}")

                if result.graph_path:
                    st.markdown("**Graph traversal**:")
                    for step in result.graph_path:
                        st.markdown(f"  - {step}")

                if result.sources:
                    with st.expander("Sources"):
                        for i, src in enumerate(result.sources):
                            st.markdown(
                                f"**[Source {i + 1}]** (score: {src.score:.3f}, via: {src.source})"
                            )
                            st.markdown(f"> {src.text[:200]}...")
                            st.divider()

                # Fact check
                fc = pipeline.fact_check(result.answer, result.sources)
                if fc.unsupported_claims:
                    st.warning(
                        f"Fact check: {fc.support_score * 100:.0f}% supported. "
                        f"{len(fc.unsupported_claims)} claim(s) not directly in sources."
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer}
                )


# -- Tab 2: Knowledge Graph --
with tab_graph:
    st.markdown("### Entity Knowledge Graph")

    kg = st.session_state.graph_pipeline.knowledge_graph

    if kg.entity_count == 0:
        st.info("No entities extracted. Try uploading a document with named entities.")
    else:
        # Render with matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        G = kg.graph
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

        type_colors = {
            "PERSON": "#4fc3f7",
            "ORG": "#81c784",
            "PRODUCT": "#ffb74d",
            "CONCEPT": "#ce93d8",
            "LOCATION": "#ef9a9a",
        }

        node_colors = [
            type_colors.get(G.nodes[n].get("entity_type", "CONCEPT"), "#e0e0e0")
            for n in G.nodes
        ]
        node_sizes = [
            300 + 100 * G.nodes[n].get("mentions", 1) for n in G.nodes
        ]
        labels = {n: G.nodes[n].get("name", n)[:20] for n in G.nodes}

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555", alpha=0.5, arrows=True)
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9
        )
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_color="white")

        ax.set_title("Knowledge Graph", color="white", fontsize=14)
        ax.axis("off")

        # Legend
        for etype, color in type_colors.items():
            ax.scatter([], [], c=color, s=100, label=etype)
        ax.legend(loc="upper left", fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")

        st.pyplot(fig)
        plt.close(fig)

        # Entity table
        st.markdown("### Entities")
        entity_data = []
        for node_id in G.nodes:
            data = G.nodes[node_id]
            entity_data.append({
                "Name": data.get("name", ""),
                "Type": data.get("entity_type", ""),
                "Mentions": data.get("mentions", 0),
                "Connected Chunks": len(data.get("source_chunks", [])),
            })
        if entity_data:
            st.dataframe(entity_data, use_container_width=True)


# -- Tab 3: Comparison --
with tab_compare:
    st.markdown("### Basic RAG vs GraphRAG")
    st.markdown("Enter a query to compare retrieval quality side-by-side.")

    compare_query = st.text_input("Comparison query", key="compare_q")

    if compare_query and st.button("Compare", key="compare_btn"):
        col_basic, col_graph = st.columns(2)

        with col_basic:
            st.markdown("#### Basic RAG (BM25)")
            basic: BasicRAGPipeline = st.session_state.basic_pipeline
            basic_result = basic.query(compare_query)
            st.markdown(basic_result.answer[:500])
            st.metric("Latency", f"{basic_result.latency_ms}ms")
            st.metric("Sources", len(basic_result.sources))

            if basic_result.sources:
                with st.expander("Sources"):
                    for i, src in enumerate(basic_result.sources):
                        st.markdown(f"**[Source {i+1}]** (score: {src.score:.3f})")
                        st.markdown(f"> {src.content}...")

        with col_graph:
            st.markdown("#### GraphRAG (BM25 + Graph)")
            graph_pipeline: GraphRAGPipeline = st.session_state.graph_pipeline
            graph_result = graph_pipeline.query(compare_query)
            st.markdown(graph_result.answer[:500])

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{graph_result.confidence * 100:.1f}%")
            c2.metric("Latency", f"{graph_result.latency_ms}ms")
            c3.metric("Sources", len(graph_result.sources))

            if graph_result.entities_found:
                st.markdown(f"**Entities**: {', '.join(graph_result.entities_found)}")

            if graph_result.graph_path:
                st.markdown("**Graph path**:")
                for step in graph_result.graph_path:
                    st.markdown(f"  - {step}")

            if graph_result.sources:
                with st.expander("Sources"):
                    for i, src in enumerate(graph_result.sources):
                        st.markdown(
                            f"**[Source {i+1}]** (score: {src.score:.3f}, via: {src.source})"
                        )
                        st.markdown(f"> {src.text[:200]}...")
