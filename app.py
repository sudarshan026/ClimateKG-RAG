import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force offline mode for Hugging Face Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config.settings import settings
from src.document_processor.store import DocumentStore
from src.document_processor.extractor import DocumentExtractor
from src.chunking.chunker import TextChunker
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.schema import VALID_NODE_TYPES, VALID_RELATION_TYPES
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridGraphRAGRetriever
from src.qa.generator import GroundedAnswerGenerator
from src.evaluation.benchmark_dataset import CLIMATE_BENCHMARK_QUERIES
from src.evaluation.evaluator import ClimateEvaluator
from src.visualization.graph_visualizer import GraphVisualizer, NODE_COLOR_MAP
from src.visualization.metrics_visualizer import MetricsVisualizer
from data.sample_climate_docs import generate_sample_corpus

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClimateKG-RAG | UN SDG 13 Knowledge Discovery",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #90A4AE;
        margin-bottom: 1.5rem;
    }
    .sdg-badge {
        background-color: #3F7E44;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .stat-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .citation-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .path-badge {
        background-color: #263238;
        border: 1px solid #455a64;
        border-radius: 4px;
        padding: 6px 10px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #80cbc4;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization & Lazy Caching
# ---------------------------------------------------------
@st.cache_resource
def get_document_store():
    settings.ensure_directories()
    return DocumentStore()

@st.cache_resource
def get_graph_store():
    return NetworkXGraphStore()

@st.cache_resource
def get_neo4j_client():
    client = Neo4jClient()
    if settings.USE_NEO4J_IF_AVAILABLE:
        client.connect()
    return client

@st.cache_resource
def get_graph_builder():
    return GraphBuilder(networkx_store=get_graph_store(), neo4j_client=get_neo4j_client())

@st.cache_resource
def get_retrieval_suite():
    doc_store = get_document_store()
    chunks = doc_store.get_all_chunks()
    nx_store = get_graph_store()
    neo_client = get_neo4j_client()

    tfidf = TFIDFRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    graph = GraphRetriever(nx_store=nx_store, neo4j_client=neo_client, max_hops=settings.GRAPH_MAX_HOPS)
    hybrid = HybridGraphRAGRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        graph_retriever=graph,
        rrf_k=settings.RRF_K
    )
    generator = GroundedAnswerGenerator()
    return {
        "tfidf": tfidf,
        "bm25": bm25,
        "dense": dense,
        "graph": graph,
        "hybrid": hybrid,
        "generator": generator
    }

doc_store = get_document_store()
graph_store = get_graph_store()
graph_builder = get_graph_builder()
neo4j_client = get_neo4j_client()
suite = get_retrieval_suite()

# ---------------------------------------------------------
# Sidebar Navigation & System Quick-Stats
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/antigravity-ai/assets/main/sdg13.png" if False else "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Sustainable_Development_Goal_13.svg/1200px-Sustainable_Development_Goal_13.svg.png", width=120)
    st.markdown("### ClimateKG-RAG")
    st.markdown("<span class='sdg-badge'>UN SDG 13: Climate Action</span>", unsafe_allow_html=True)
    st.markdown("---")

    nav_options = [
        "1. Home / Overview",
        "2. Document Management",
        "3. Knowledge Graph Explorer",
        "4. Climate Question Answering",
        "5. Retrieval Comparison",
        "6. Evaluation Dashboard",
        "7. System Configuration"
    ]
    default_nav_idx = 0
    if "nav_target" in st.session_state and st.session_state["nav_target"] in nav_options:
        default_nav_idx = nav_options.index(st.session_state["nav_target"])
        del st.session_state["nav_target"]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=default_nav_idx
    )

    st.markdown("---")
    st.markdown("#### Live System Telemetry")
    live_doc_stats = doc_store.get_statistics()
    live_g_stats = graph_store.get_statistics()
    st.metric("Indexed Documents", live_doc_stats["total_documents"])
    st.metric("Total Chunks", live_doc_stats["total_chunks"])
    st.metric("Graph Entities", live_g_stats["total_nodes"])
    st.metric("Graph Relations", live_g_stats["total_edges"])

    neo_status = "Connected" if neo4j_client.is_connected() else "Standalone / Fallback"
    st.caption(f"**Neo4j:** {neo_status}")

# ---------------------------------------------------------
# PAGE 1: HOME / OVERVIEW
# ---------------------------------------------------------
if selected_page == "1. Home / Overview":
    st.markdown("<div class='main-header'>🌍 ClimateKG-RAG: Knowledge Discovery System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>A Hybrid Knowledge Graph and Multi-Paradigm Information Retrieval System for Climate Change Evidence Grounding</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <h4>Indexed Docs</h4>
            <h2>{live_doc_stats['total_documents']}</h2>
            <small>{live_doc_stats['total_pages']} Total Pages</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <h4>Processed Chunks</h4>
            <h2>{live_doc_stats['total_chunks']}</h2>
            <small>{live_doc_stats['total_words']} Extracted Words</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <h4>KG Entities</h4>
            <h2>{live_g_stats['total_nodes']}</h2>
            <small>{len(live_g_stats['nodes_by_type'])} Distinct Node Types</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <h4>Evidence Edges</h4>
            <h2>{live_g_stats['total_edges']}</h2>
            <small>Density: {live_g_stats['density']}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### System Architecture & Pipeline")
    st.info("""
    **ClimateKG-RAG Architecture:**
    1. **Corpus Ingestion & Chunking:** PyMuPDF, python-docx, and TXT extractors with sentence-aware chunking and SHA-256 deduplication.
    2. **Knowledge Graph Engine:** Climate schema with 10 Node Types and 8 Relation Types, dual-persisted across NetworkX and Neo4j Community.
    3. **Multi-Paradigm Retrieval:** TF-IDF, BM25 Okapi, Dense Semantic (`all-MiniLM-L6-v2`), and Multi-Hop Graph Traversal.
    4. **Hybrid GraphRAG Fusion:** Reciprocal Rank Fusion (RRF) combining lexical, dense, and structural relational graph pathways.
    5. **Evidence Grounding:** Citation-attributed answer synthesis strictly grounded in verified excerpts to eliminate hallucination.
    """)

    st.markdown("### Quick Setup Actions")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("📥 Load & Ingest Sample Climate Corpus", use_container_width=True):
            with st.spinner("Generating sample corpus and building graph..."):
                generate_sample_corpus()
                raw_files = list(settings.RAW_DATA_DIR.glob("*.txt"))
                chunker = TextChunker(chunk_size=settings.DEFAULT_CHUNK_SIZE, chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP)
                for fpath in raw_files:
                    doc = DocumentExtractor.extract(fpath, fpath.name)
                    chunks = chunker.chunk_document(doc)
                    doc_store.add_document(doc, chunks)
                all_chunks = doc_store.get_all_chunks()
                graph_builder.process_chunks(all_chunks)
                st.cache_resource.clear()
                st.success("Successfully ingested 8 authentic climate science documents and built Knowledge Graph!")
                st.rerun()

    with q_col2:
        if st.button("🔬 Jump to Climate Question Answering", use_container_width=True):
            st.session_state["nav_target"] = "4. Climate Question Answering"
            st.rerun()

    with q_col3:
        if st.button("📊 Run Evaluation Benchmark", use_container_width=True):
            st.session_state["nav_target"] = "6. Evaluation Dashboard"
            st.rerun()

# ---------------------------------------------------------
# PAGE 2: DOCUMENT MANAGEMENT
# ---------------------------------------------------------
elif selected_page == "2. Document Management":
    st.markdown("## 📁 Climate Document Management")
    st.caption("Upload, inspect, chunk, and manage climate research publications, reports, and whitepapers.")

    upload_col, settings_col = st.columns([2, 1])

    with settings_col:
        st.markdown("#### Chunking Parameters")
        chunk_size = st.slider("Chunk Size (words)", min_value=100, max_value=1000, value=settings.DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk Overlap (words)", min_value=20, max_value=300, value=settings.DEFAULT_CHUNK_OVERLAP, step=20)
        st.caption("Sentence boundary awareness preserves complete scientific assertions within chunks.")

    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload Climate Documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button(f"Process & Ingest {len(uploaded_files)} File(s)", type="primary"):
                progress_bar = st.progress(0)
                chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                ingested_count = 0

                for idx, file in enumerate(uploaded_files):
                    file_bytes = file.read()
                    doc = DocumentExtractor.extract(file_bytes, file.name)
                    chunks = chunker.chunk_document(doc)
                    added = doc_store.add_document(doc, chunks)
                    if added:
                        ingested_count += 1
                        # Automatically update Knowledge Graph
                        graph_builder.process_chunks(chunks)
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                st.cache_resource.clear()
                st.success(f"Successfully processed {ingested_count} new documents into DocumentStore & Knowledge Graph!")
                st.rerun()

    st.markdown("---")
    st.markdown("### Currently Ingested Documents")
    all_docs = doc_store.get_all_documents()

    if not all_docs:
        st.info("No documents currently ingested. Click below or upload documents above.")
        if st.button("Ingest Default Climate Corpus"):
            generate_sample_corpus()
            for fpath in settings.RAW_DATA_DIR.glob("*.txt"):
                doc = DocumentExtractor.extract(fpath, fpath.name)
                chunks = TextChunker().chunk_document(doc)
                doc_store.add_document(doc, chunks)
            graph_builder.process_chunks(doc_store.get_all_chunks())
            st.cache_resource.clear()
            st.rerun()
    else:
        doc_rows = []
        for d in all_docs:
            c_count = len(doc_store.get_chunks_for_document(d.doc_id))
            doc_rows.append({
                "Document ID": d.doc_id,
                "Filename": d.filename,
                "Title": d.title,
                "Type": d.file_type.upper(),
                "Pages": d.total_pages,
                "Chunks": c_count,
                "SHA-256": d.file_hash[:12] + "..."
            })
        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

        st.markdown("#### Document Inspector & Chunks Viewer")
        selected_doc_id = st.selectbox("Select Document to Inspect:", [d.doc_id for d in all_docs], format_func=lambda x: f"{x} - {doc_store.get_document(x).filename}")
        if selected_doc_id:
            inspected_doc = doc_store.get_document(selected_doc_id)
            doc_chunks = doc_store.get_chunks_for_document(selected_doc_id)

            c_info, c_action = st.columns([3, 1])
            with c_info:
                st.write(f"**Title:** {inspected_doc.title}")
                st.write(f"**Filename:** `{inspected_doc.filename}` | **Pages:** {inspected_doc.total_pages} | **Total Chunks:** {len(doc_chunks)}")
            with c_action:
                if st.button("🗑️ Delete Document", key=f"del_{selected_doc_id}", type="secondary"):
                    doc_store.delete_document(selected_doc_id)
                    graph_builder.remove_document(selected_doc_id)
                    st.cache_resource.clear()
                    st.success(f"Deleted {inspected_doc.filename} and synchronized Knowledge Graph & Retrieval indices.")
                    st.rerun()

            with st.expander(f"View {len(doc_chunks)} Processed Chunks"):
                for chk in doc_chunks:
                    st.markdown(f"**Chunk ID:** `{chk.chunk_id}` | Page {chk.page_number} ({chk.word_count} words)")
                    st.text_area("", value=chk.text, height=100, key=f"txt_{chk.chunk_id}", disabled=True)

# ---------------------------------------------------------
# PAGE 3: KNOWLEDGE GRAPH EXPLORER
# ---------------------------------------------------------
elif selected_page == "3. Knowledge Graph Explorer":
    st.markdown("## 🕸️ Climate Knowledge Graph Explorer")
    st.caption("Interactive visualization, entity relationships, multi-hop pathways, and evidence provenance.")

    g_stats = graph_store.get_statistics()
    if g_stats["total_nodes"] == 0:
        st.warning("Knowledge Graph is currently empty. Ingest documents in Tab 2 to populate.")
    else:
        # Schema color legend
        st.markdown("##### Entity Schema & Type Legend")
        legend_cols = st.columns(5)
        for idx, (etype, color) in enumerate(NODE_COLOR_MAP.items()):
            col = legend_cols[idx % 5]
            col.markdown(f"<span style='color:{color}; font-weight:bold;'>●</span> {etype}", unsafe_allow_html=True)

        st.markdown("---")
        filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1.5, 1])

        with filter_col1:
            all_types = list(VALID_NODE_TYPES)
            selected_types = st.multiselect("Filter by Entity Type:", all_types, default=all_types[:6])

        with filter_col2:
            search_query = st.text_input("Search Entity Name:", placeholder="e.g. India, Global Warming, Solar...")

        with filter_col3:
            max_viz_nodes = st.slider("Max Nodes in Visualizer", 20, 150, 60)

        # Filter subgraph
        sub_graph = graph_store.graph.copy()
        if selected_types:
            nodes_to_keep = [n for n, d in sub_graph.nodes(data=True) if d.get("type") in selected_types]
            sub_graph = sub_graph.subgraph(nodes_to_keep).copy()

        if search_query:
            matched_nodes = [n for n, d in sub_graph.nodes(data=True) if search_query.lower() in d.get("name", "").lower()]
            # Include 1-hop neighbors
            expanded = set(matched_nodes)
            for n in matched_nodes:
                expanded.update(sub_graph.neighbors(n))
            sub_graph = sub_graph.subgraph(list(expanded)).copy()

        # Render PyVis Interactive Network
        st.markdown(f"#### Interactive Graph Visualization ({sub_graph.number_of_nodes()} Nodes, {sub_graph.number_of_edges()} Edges)")
        if sub_graph.number_of_nodes() > 0:
            pyvis_html = GraphVisualizer.generate_pyvis_html(sub_graph, height="550px", max_nodes=max_viz_nodes)
            components.html(pyvis_html, height=580)
        else:
            st.info("No nodes match the selected filters.")

        # Multi-Hop Traversal Sandbox
        st.markdown("---")
        st.markdown("### 🔀 Multi-Hop Relational Traversal Sandbox")
        st.caption("Inspect how climate phenomena cascade into regional impacts and adaptation pathways across graph hops.")

        all_nodes_dict = {nid: graph_store.graph.nodes[nid].get("name", nid) for nid in graph_store.graph.nodes}
        sorted_nodes = sorted(all_nodes_dict.items(), key=lambda x: x[1])

        hop_c1, hop_c2 = st.columns([2, 1])
        with hop_c1:
            start_node_tuple = st.selectbox(
                "Select Focal Entity Node:",
                options=sorted_nodes,
                format_func=lambda x: f"{x[1]} ({graph_store.graph.nodes[x[0]].get('type', 'Entity')})"
            )
        with hop_c2:
            hops_count = st.slider("Traversal Depth (Hops)", min_value=1, max_value=3, value=2)

        if start_node_tuple:
            start_nid = start_node_tuple[0]
            discovered_paths = graph_store.multi_hop_traversal(start_nid, max_hops=hops_count)

            st.write(f"**Discovered {len(discovered_paths)} multi-hop relational pathways:**")
            for idx, p in enumerate(discovered_paths[:8], start=1):
                p_nodes = p["nodes"]
                p_edges = p["edges"]
                path_str = f"**Path {idx}:** " + " ➔ ".join([f"`{n.get('name')}`" for n in p_nodes])
                st.markdown(path_str)
                with st.expander(f"View Evidence for Path {idx} ({p['hops']} Hop{'s' if p['hops']>1 else ''})"):
                    for e in p_edges:
                        st.markdown(f"- **Relation:** `{e.get('type')}`")
                        st.markdown(f"  **Evidence Quote:** *\"{e.get('evidence_text')}\"*")
                        st.caption(f"Source: `{e.get('filename')}` | Page {e.get('page_number')} | Chunk: `{e.get('source_chunk_id')}`")

# ---------------------------------------------------------
# PAGE 4: CLIMATE QUESTION ANSWERING
# ---------------------------------------------------------
elif selected_page == "4. Climate Question Answering":
    st.markdown("## 💬 Evidence-Grounded Climate Question Answering")
    st.caption("Ask natural language climate science questions and receive answers strictly verified with citations and graph pathways.")

    # Preset Sample Queries
    sample_queries = [
        "How does global warming affect agriculture in India, and what adaptation strategies are documented?",
        "What causal relationship exists between greenhouse gas emissions and extreme weather events?",
        "What adaptation strategies does the Indian Council of Agricultural Research (ICAR) recommend for drought resilience?",
        "How does carbon dioxide emission lead to agricultural yield decline in India, and what adaptation strategy is documented?",
        "What global temperature threshold does the Paris Agreement aim to hold global warming well below?",
        "What climate phenomenon causes sea level rise?"
    ]

    selected_sample = st.selectbox("Sample Climate Research Questions (or enter custom below):", ["-- Select a preset question --"] + sample_queries)

    user_query = st.text_input(
        "Enter Climate Research Query:",
        value=selected_sample if selected_sample != "-- Select a preset question --" else "",
        placeholder="e.g. How does climate change impact water scarcity in South Asia?"
    )

    retrieval_col, topk_col, exec_col = st.columns([2, 1, 1])
    with retrieval_col:
        method_choice = st.selectbox(
            "Select Information Retrieval Method:",
            [
                "Hybrid GraphRAG (BM25 + Dense + KG)",
                "Knowledge Graph Multi-Hop Traversal",
                "Dense Semantic Retrieval (all-MiniLM-L6-v2)",
                "BM25 Lexical Retrieval",
                "TF-IDF Lexical Retrieval"
            ]
        )
    with topk_col:
        top_k = st.number_input("Top-K Passages", min_value=1, max_value=10, value=settings.TOP_K_RETRIEVAL)
    with exec_col:
        st.write("")
        st.write("")
        run_qa_button = st.button("🚀 Answer Question", type="primary", use_container_width=True)

    if run_qa_button and user_query.strip():
        with st.spinner("Executing retrieval pipeline and synthesizing grounded response..."):
            # Map choice to retriever
            method_key = "hybrid"
            if "Knowledge Graph" in method_choice:
                method_key = "graph"
            elif "Dense" in method_choice:
                method_key = "dense"
            elif "BM25" in method_choice:
                method_key = "bm25"
            elif "TF-IDF" in method_choice:
                method_key = "tfidf"

            active_retriever = suite[method_key]
            retrieval_resp = active_retriever.retrieve(user_query, top_k=top_k)
            qa_resp = suite["generator"].answer_query(
                query=user_query,
                retrieval_method=retrieval_resp.method,
                results=retrieval_resp.results
            )

            # Display Answer Card
            st.markdown("---")
            ans_header_col1, ans_header_col2, ans_header_col3 = st.columns([2, 1, 1])
            with ans_header_col1:
                st.markdown("### 📝 Grounded Scientific Answer")
            with ans_header_col2:
                conf_color = "#388E3C" if qa_resp.confidence == "High" else "#F57C00" if qa_resp.confidence == "Medium" else "#D32F2F"
                st.markdown(f"<span style='background-color:{conf_color}; color:white; padding:4px 10px; border-radius:12px; font-weight:600;'>Confidence: {qa_resp.confidence}</span>", unsafe_allow_html=True)
            with ans_header_col3:
                st.markdown(f"⏱️ **Latency:** `{retrieval_resp.latency_ms + qa_resp.latency_ms:.1f} ms`")

            st.info(f"**Synthesized by:** {qa_resp.generator_type} | **Method:** {retrieval_resp.method}")
            st.markdown(qa_resp.answer)

            # Relational Graph Paths
            if qa_resp.graph_paths:
                st.markdown("#### 🔗 Knowledge Graph Discovered Pathways")
                for p in qa_resp.graph_paths:
                    st.markdown(f"<div class='path-badge'>{p}</div>", unsafe_allow_html=True)

            # Supporting Evidence & Citations
            st.markdown("#### 📚 Verified Document Citations & Passages")
            for cit in qa_resp.citations:
                with st.expander(f"[{cit.source_index}] Source: {cit.filename} (Page {cit.page_number}, Chunk: {cit.chunk_id})"):
                    st.write(f"**Document Title:** {cit.doc_title}")
                    st.markdown(f"*\"{cit.snippet}\"*")

            # Retrieved Results Table
            st.markdown("#### 📊 Ranked Retrieval Breakdown")
            res_data = []
            for r in retrieval_resp.results:
                res_data.append({
                    "Rank": r.rank,
                    "Score": r.score,
                    "Document": r.filename,
                    "Page": r.page_number,
                    "Chunk ID": r.chunk_id,
                    "Method": r.method
                })
            st.dataframe(pd.DataFrame(res_data), use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: INFORMATION RETRIEVAL COMPARISON
# ---------------------------------------------------------
elif selected_page == "5. Retrieval Comparison":
    st.markdown("## ⚖️ Information Retrieval Multi-Paradigm Comparison")
    st.caption("Compare TF-IDF, BM25, Dense Semantic, Knowledge Graph, and Hybrid GraphRAG on identical queries in real time.")

    comp_query = st.text_input(
        "Enter Comparison Query:",
        value="How does global warming cause agricultural yield decline in India and what adaptation strategies exist?",
        placeholder="Enter query to compare..."
    )

    if st.button("Run Multi-Retriever Comparison", type="primary"):
        with st.spinner("Executing all 5 retrieval pipelines..."):
            retrievers = {
                "TF-IDF": suite["tfidf"],
                "BM25": suite["bm25"],
                "Dense Semantic": suite["dense"],
                "Knowledge Graph": suite["graph"],
                "Hybrid GraphRAG": suite["hybrid"]
            }

            comparison_results = {}
            for name, ret in retrievers.items():
                comparison_results[name] = ret.retrieve(comp_query, top_k=3)

            # Comparison Metrics Table
            table_rows = []
            for name, resp in comparison_results.items():
                top_item = resp.results[0] if resp.results else None
                table_rows.append({
                    "Method": name,
                    "Latency (ms)": resp.latency_ms,
                    "Total Searched": resp.total_chunks_searched,
                    "Top-1 Score": top_item.score if top_item else 0.0,
                    "Top-1 Source": f"{top_item.filename} (P.{top_item.page_number})" if top_item else "None",
                    "Top-1 Snippet": (top_item.text[:90] + "...") if top_item else "No results"
                })

            st.markdown("### Comparative Performance Summary")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Side-by-side Top Results View
            st.markdown("### Top Retrieved Excerpt by Method")
            tabs = st.tabs(list(retrievers.keys()))
            for idx, (name, resp) in enumerate(comparison_results.items()):
                with tabs[idx]:
                    st.write(f"**Execution Latency:** `{resp.latency_ms} ms`")
                    if resp.results:
                        for rank, r in enumerate(resp.results, start=1):
                            st.markdown(f"**Rank {rank} (Score: {r.score})** — `{r.filename}`, Page {r.page_number}")
                            st.info(r.text)
                    else:
                        st.warning("No results retrieved by this method.")

# ---------------------------------------------------------
# PAGE 6: EVALUATION DASHBOARD
# ---------------------------------------------------------
elif selected_page == "6. Evaluation Dashboard":
    st.markdown("## 📈 Quantitative Evaluation & Benchmark Dashboard")
    st.caption("Standardized evaluation across 32 verifiable climate queries comparing Precision@K, Recall@K, MRR, MAP, NDCG@K, Latency, and Faithfulness.")

    eval_control_c1, eval_control_c2 = st.columns([2, 1])
    with eval_control_c1:
        st.markdown(f"**Benchmark Dataset:** `{len(CLIMATE_BENCHMARK_QUERIES)}` Verified Climate Science Queries")
        st.write("Categories: Factual, Semantic, Relational, Multi-Hop, Adaptation/Mitigation Strategies.")

    with eval_control_c2:
        run_full_eval = st.button("▶️ Run Full 32-Query Benchmark", type="primary", use_container_width=True)

    summary_file = settings.EVALUATION_DIR / "evaluation_summary.csv"

    if run_full_eval:
        with st.spinner("Running full 32-query comparative benchmark across 5 retrieval algorithms..."):
            evaluator = ClimateEvaluator(
                tfidf_retriever=suite["tfidf"],
                bm25_retriever=suite["bm25"],
                dense_retriever=suite["dense"],
                graph_retriever=suite["graph"],
                hybrid_retriever=suite["hybrid"],
                generator=suite["generator"]
            )
            eval_result = evaluator.run_comparative_evaluation(top_k=settings.TOP_K_RETRIEVAL, compute_faithfulness=True)
            st.success("Benchmark evaluation completed and exported to CSV!")

    # Display Computed Results if summary file exists
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        st.markdown("### Actual Computed Benchmark Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### Visual Performance Comparison")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_bar = MetricsVisualizer.create_metrics_bar_chart(summary_df)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            fig_radar = MetricsVisualizer.create_radar_chart(summary_df)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### Latency Profile")
        fig_latency = MetricsVisualizer.create_latency_bar_chart(summary_df)
        st.plotly_chart(fig_latency, use_container_width=True)

        # CSV Export Buttons
        st.markdown("### Export Experiment Data")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with open(summary_file, "rb") as f:
                st.download_button("📥 Download Summary CSV", f, file_name="climatekg_evaluation_summary.csv", mime="text/csv", use_container_width=True)
        with exp_col2:
            detailed_file = settings.EVALUATION_DIR / "evaluation_detailed.csv"
            if detailed_file.exists():
                with open(detailed_file, "rb") as f:
                    st.download_button("📥 Download Detailed Per-Query CSV", f, file_name="climatekg_evaluation_detailed.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No benchmark results found on disk. Click 'Run Full 32-Query Benchmark' above to execute.")

# ---------------------------------------------------------
# PAGE 7: SYSTEM CONFIGURATION
# ---------------------------------------------------------
elif selected_page == "7. System Configuration":
    st.markdown("## ⚙️ System Configuration & Diagnostics")
    st.caption("Inspect and manage environment variables, database connectors, and local model backends.")

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("### Neo4j Community Graph Database")
        st.text_input("Neo4j Bolt URI", value=settings.NEO4J_URI, disabled=True)
        st.text_input("Neo4j User", value=settings.NEO4J_USER, disabled=True)
        st.text_input("Neo4j Database", value=settings.NEO4J_DATABASE, disabled=True)

        neo_conn = neo4j_client.is_connected()
        if neo_conn:
            st.success("● Neo4j Connection Active (bolt://localhost:7687)")
        else:
            st.warning("● Neo4j Offline — NetworkX Persistent In-Memory Fallback Active")
            st.caption("To run Neo4j Community via Docker: `docker compose up -d`")

        if st.button("Test Neo4j Connection"):
            conn_success = neo4j_client.connect()
            if conn_success:
                st.success("Connection test successful!")
            else:
                st.error("Could not reach Neo4j server on configured port.")

    with cfg_col2:
        st.markdown("### Local Ollama LLM Backend")
        st.text_input("Ollama Base URL", value=settings.OLLAMA_BASE_URL, disabled=True)
        st.text_input("Ollama Model", value=settings.OLLAMA_MODEL, disabled=True)

        ollama_active = suite["generator"].is_ollama_available()
        if ollama_active:
            st.success(f"● Ollama Online ({settings.OLLAMA_MODEL})")
        else:
            st.info("● Ollama Offline — Deterministic Extractive Synthesis Engine Active")
            st.caption("Start local Ollama with `ollama run llama3` for full LLM generative features.")

        if st.button("Test Ollama Connection"):
            if suite["generator"].is_ollama_available():
                st.success("Ollama connection verified!")
            else:
                st.error("Ollama endpoint not responding.")

    st.markdown("---")
    st.markdown("### Storage & Cache Diagnostics")
    cache_c1, cache_c2, cache_c3 = st.columns(3)
    emb_file = settings.VECTOR_CACHE_DIR / "embeddings.npz"
    emb_size_kb = round(emb_file.stat().st_size / 1024, 1) if emb_file.exists() else 0
    cache_c1.metric("Embedding Cache Size", f"{emb_size_kb} KB")

    g_file = settings.GRAPH_STORAGE_DIR / "graph.json"
    g_size_kb = round(g_file.stat().st_size / 1024, 1) if g_file.exists() else 0
    cache_c2.metric("Knowledge Graph File", f"{g_size_kb} KB")

    docs_file = settings.PROCESSED_DATA_DIR / "documents.json"
    docs_size_kb = round(docs_file.stat().st_size / 1024, 1) if docs_file.exists() else 0
    cache_c3.metric("Document Index File", f"{docs_size_kb} KB")

