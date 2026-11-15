import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Set offline mode for Hugging Face
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from config.settings import settings
from src.document_processor.store import DocumentStore
from src.document_processor.extractor import DocumentExtractor
from src.chunking.chunker import TextChunker, DocumentChunk
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
from src.visualization.graph_visualizer import NODE_COLOR_MAP
from data.sample_climate_docs import generate_sample_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("climatekg-api")

app = FastAPI(
    title="ClimateKG-RAG API",
    description="Backend API for Climate Change Knowledge Graph & Multi-Paradigm IR System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global System State & Singletons
# -----------------------------------------------------------------------------
class SystemManager:
    def __init__(self):
        settings.ensure_directories()
        self.doc_store = DocumentStore()
        self.graph_store = NetworkXGraphStore()
        self.neo4j_client = Neo4jClient()
        if settings.USE_NEO4J_IF_AVAILABLE:
            try:
                self.neo4j_client.connect()
            except Exception as e:
                logger.warning(f"Neo4j connection error on init: {e}")
        self.graph_builder = GraphBuilder(networkx_store=self.graph_store, neo4j_client=self.neo4j_client)
        self.generator = GroundedAnswerGenerator()
        
        self.tfidf = None
        self.bm25 = None
        self.dense = None
        self.graph = None
        self.hybrid = None
        self.rebuild_retrievers()

    def rebuild_retrievers(self):
        chunks = self.doc_store.get_all_chunks()
        logger.info(f"Rebuilding retrievers with {len(chunks)} chunks...")
        self.tfidf = TFIDFRetriever(chunks)
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)
        self.graph = GraphRetriever(
            nx_store=self.graph_store,
            neo4j_client=self.neo4j_client,
            max_hops=settings.GRAPH_MAX_HOPS
        )
        self.hybrid = HybridGraphRAGRetriever(
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            graph_retriever=self.graph,
            rrf_k=settings.RRF_K
        )

sys_mgr = SystemManager()

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class QARequest(BaseModel):
    query: str
    method: str = "hybrid"  # "hybrid", "graph", "dense", "bm25", "tfidf"
    top_k: int = 5

class CompareRequest(BaseModel):
    query: str
    top_k: int = 3

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/stats")
def get_system_stats():
    doc_stats = sys_mgr.doc_store.get_statistics()
    graph_stats = sys_mgr.graph_store.get_statistics()
    
    emb_file = settings.VECTOR_CACHE_DIR / "embeddings.npz"
    emb_kb = round(emb_file.stat().st_size / 1024, 1) if emb_file.exists() else 0
    
    g_file = settings.GRAPH_STORAGE_DIR / "graph.json"
    g_kb = round(g_file.stat().st_size / 1024, 1) if g_file.exists() else 0
    
    return {
        "documents": doc_stats["total_documents"],
        "pages": doc_stats["total_pages"],
        "chunks": doc_stats["total_chunks"],
        "words": doc_stats["total_words"],
        "nodes": graph_stats["total_nodes"],
        "edges": graph_stats["total_edges"],
        "density": graph_stats["density"],
        "nodes_by_type": graph_stats["nodes_by_type"],
        "edges_by_type": graph_stats["edges_by_type"],
        "neo4j_connected": sys_mgr.neo4j_client.is_connected(),
        "ollama_available": sys_mgr.generator.is_ollama_available(),
        "embedding_cache_kb": emb_kb,
        "graph_file_kb": g_kb,
        "supported_node_types": list(VALID_NODE_TYPES),
        "supported_relation_types": list(VALID_RELATION_TYPES),
        "node_color_map": NODE_COLOR_MAP
    }

@app.get("/api/documents")
def list_documents():
    docs = sys_mgr.doc_store.get_all_documents()
    res = []
    for d in docs:
        c_count = len(sys_mgr.doc_store.get_chunks_for_document(d.doc_id))
        res.append({
            "doc_id": d.doc_id,
            "filename": d.filename,
            "title": d.title,
            "file_type": d.file_type,
            "total_pages": d.total_pages,
            "chunk_count": c_count,
            "file_hash": d.file_hash,
            "metadata": d.metadata
        })
    return res

@app.get("/api/documents/{doc_id}")
def get_document_details(doc_id: str):
    doc = sys_mgr.doc_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = sys_mgr.doc_store.get_chunks_for_document(doc_id)
    return {
        "document": {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "title": doc.title,
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "file_hash": doc.file_hash,
            "metadata": doc.metadata
        },
        "chunks": [c.to_dict() for c in chunks]
    }

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    doc = sys_mgr.doc_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    sys_mgr.doc_store.delete_document(doc_id)
    removed_edges = sys_mgr.graph_builder.remove_document(doc_id)
    sys_mgr.rebuild_retrievers()
    
    return {
        "success": True,
        "message": f"Deleted document {doc.filename}. Pruned {removed_edges} graph edges and updated indices."
    }

@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(settings.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(settings.DEFAULT_CHUNK_OVERLAP)
):
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ingested_count = 0
    all_new_chunks = []

    for file in files:
        content = await file.read()
        try:
            doc = DocumentExtractor.extract(content, file.filename)
            chunks = chunker.chunk_document(doc)
            added = sys_mgr.doc_store.add_document(doc, chunks)
            if added:
                ingested_count += 1
                all_new_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process uploaded file {file.filename}: {e}")

    if all_new_chunks:
        sys_mgr.graph_builder.process_chunks(all_new_chunks)
        sys_mgr.rebuild_retrievers()

    return {
        "success": True,
        "ingested_count": ingested_count,
        "chunks_created": len(all_new_chunks),
        "total_documents": len(sys_mgr.doc_store.get_all_documents())
    }

@app.post("/api/documents/ingest-default")
def ingest_default_corpus():
    generate_sample_corpus()
    raw_files = list(settings.RAW_DATA_DIR.glob("*.txt"))
    chunker = TextChunker(chunk_size=settings.DEFAULT_CHUNK_SIZE, chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP)
    
    ingested = 0
    for fpath in raw_files:
        doc = DocumentExtractor.extract(fpath, fpath.name)
        chunks = chunker.chunk_document(doc)
        if sys_mgr.doc_store.add_document(doc, chunks):
            ingested += 1
            
    all_chunks = sys_mgr.doc_store.get_all_chunks()
    sys_mgr.graph_builder.process_chunks(all_chunks)
    sys_mgr.rebuild_retrievers()
    
    return {
        "success": True,
        "documents_ingested": ingested,
        "total_documents": len(sys_mgr.doc_store.get_all_documents()),
        "total_chunks": len(all_chunks)
    }

@app.get("/api/graph")
def get_graph_data(
    node_types: Optional[List[str]] = Query(None),
    search: Optional[str] = Query(None),
    max_nodes: int = Query(100)
):
    g = sys_mgr.graph_store.graph
    
    # Filter nodes
    target_nodes = list(g.nodes(data=True))
    if node_types:
        target_nodes = [n for n in target_nodes if n[1].get("type") in node_types]
        
    if search and search.strip():
        q_lower = search.strip().lower()
        matched = [n[0] for n in target_nodes if q_lower in n[1].get("name", "").lower()]
        expanded = set(matched)
        for m in matched:
            expanded.update(g.neighbors(m))
        target_nodes = [n for n in g.nodes(data=True) if n[0] in expanded]

    # Limit to max_nodes
    if len(target_nodes) > max_nodes:
        # Sort by degree
        target_nodes = sorted(target_nodes, key=lambda n: g.degree(n[0]), reverse=True)[:max_nodes]

    node_ids = set([n[0] for n in target_nodes])
    sub_g = g.subgraph(list(node_ids))

    vis_nodes = []
    for nid, data in sub_g.nodes(data=True):
        ntype = data.get("type", "Other")
        color = NODE_COLOR_MAP.get(ntype, "#78909C")
        deg = sub_g.degree(nid)
        vis_nodes.append({
            "id": nid,
            "label": data.get("name", nid),
            "type": ntype,
            "color": color,
            "size": 14 + min(deg * 3, 24),
            "degree": deg,
            "canonical_name": data.get("canonical_name", "")
        })

    vis_edges = []
    for u, v, k, data in sub_g.edges(keys=True, data=True):
        vis_edges.append({
            "id": f"{u}_{v}_{k}",
            "from": u,
            "to": v,
            "label": data.get("type", "RELATED_TO"),
            "evidence_text": data.get("evidence_text", ""),
            "doc_id": data.get("doc_id", ""),
            "filename": data.get("filename", ""),
            "page_number": data.get("page_number", 1),
            "source_chunk_id": data.get("source_chunk_id", ""),
            "confidence": data.get("confidence", 1.0)
        })

    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "total_graph_nodes": g.number_of_nodes(),
        "total_graph_edges": g.number_of_edges()
    }

@app.get("/api/graph/traverse")
def traverse_graph(node_id: str, hops: int = 2):
    if not sys_mgr.graph_store.graph.has_node(node_id):
        raise HTTPException(status_code=404, detail="Entity node not found")
    paths = sys_mgr.graph_store.multi_hop_traversal(node_id, max_hops=hops)
    return {
        "node_id": node_id,
        "node_name": sys_mgr.graph_store.graph.nodes[node_id].get("name", node_id),
        "hops": hops,
        "total_paths": len(paths),
        "paths": paths
    }

@app.get("/api/graph/path")
def find_causal_path(source_id: str, target_id: str):
    import networkx as nx
    g = sys_mgr.graph_store.graph
    if not g.has_node(source_id) or not g.has_node(target_id):
        raise HTTPException(status_code=404, detail="Source or target node not found")
    
    try:
        # Undirected path finding to allow multi-directional causal reasoning
        u_graph = g.to_undirected()
        shortest = nx.shortest_path(u_graph, source=source_id, target=target_id)
        
        path_details = []
        for i in range(len(shortest) - 1):
            n1 = shortest[i]
            n2 = shortest[i+1]
            edges = g.get_edge_data(n1, n2) or g.get_edge_data(n2, n1) or {}
            edge_info = list(edges.values())[0] if edges else {}
            path_details.append({
                "from": {"id": n1, "name": g.nodes[n1].get("name", n1), "type": g.nodes[n1].get("type")},
                "to": {"id": n2, "name": g.nodes[n2].get("name", n2), "type": g.nodes[n2].get("type")},
                "relation": edge_info.get("type", "RELATED_TO"),
                "evidence": edge_info.get("evidence_text", ""),
                "filename": edge_info.get("filename", ""),
                "page": edge_info.get("page_number", 1)
            })
            
        return {
            "found": True,
            "hops": len(shortest) - 1,
            "node_chain": [g.nodes[n].get("name", n) for n in shortest],
            "details": path_details
        }
    except nx.NetworkXNoPath:
        return {
            "found": False,
            "message": "No connected path between these two climate entities in current Knowledge Graph."
        }

@app.post("/api/qa")
def answer_question(req: QARequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    method_map = {
        "hybrid": sys_mgr.hybrid,
        "graph": sys_mgr.graph,
        "dense": sys_mgr.dense,
        "bm25": sys_mgr.bm25,
        "tfidf": sys_mgr.tfidf
    }
    retriever = method_map.get(req.method.lower(), sys_mgr.hybrid)
    
    start_t = time.perf_counter()
    retrieval_resp = retriever.retrieve(req.query, top_k=req.top_k)
    qa_resp = sys_mgr.generator.answer_query(
        query=req.query,
        retrieval_method=retrieval_resp.method,
        results=retrieval_resp.results
    )
    total_latency = (time.perf_counter() - start_t) * 1000

    return {
        "query": req.query,
        "method": retrieval_resp.method,
        "answer": qa_resp.answer,
        "confidence": qa_resp.confidence,
        "confidence_score": qa_resp.confidence_score,
        "generator_type": qa_resp.generator_type,
        "retrieval_latency_ms": retrieval_resp.latency_ms,
        "qa_latency_ms": qa_resp.latency_ms,
        "total_latency_ms": round(total_latency, 2),
        "citations": [c.to_dict() for c in qa_resp.citations],
        "graph_paths": qa_resp.graph_paths,
        "results": [r.to_dict() for r in retrieval_resp.results]
    }

@app.post("/api/compare")
def compare_retrievers(req: CompareRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrievers = {
        "Hybrid GraphRAG": sys_mgr.hybrid,
        "Dense Semantic": sys_mgr.dense,
        "BM25": sys_mgr.bm25,
        "TF-IDF": sys_mgr.tfidf,
        "Knowledge Graph": sys_mgr.graph
    }

    comparison = {}
    for name, ret in retrievers.items():
        resp = ret.retrieve(req.query, top_k=req.top_k)
        comparison[name] = {
            "method": name,
            "latency_ms": resp.latency_ms,
            "total_searched": resp.total_chunks_searched,
            "results": [r.to_dict() for r in resp.results]
        }

    return {
        "query": req.query,
        "comparison": comparison
    }

@app.get("/api/benchmark/summary")
def get_benchmark_summary():
    summary_path = settings.EVALUATION_DIR / "evaluation_summary.csv"
    if not summary_path.exists():
        return {"exists": False, "data": []}
    
    df = pd.read_csv(summary_path)
    # Convert numpy types to native Python
    records = df.to_dict(orient="records")
    return {"exists": True, "data": records}

@app.post("/api/benchmark/run")
def run_benchmark():
    evaluator = ClimateEvaluator(
        tfidf_retriever=sys_mgr.tfidf,
        bm25_retriever=sys_mgr.bm25,
        dense_retriever=sys_mgr.dense,
        graph_retriever=sys_mgr.graph,
        hybrid_retriever=sys_mgr.hybrid,
        generator=sys_mgr.generator
    )
    results = evaluator.run_comparative_evaluation(top_k=settings.TOP_K_RETRIEVAL, compute_faithfulness=True)
    summary_df = results["summary_df"]
    return {
        "success": True,
        "message": f"Evaluated 32 climate queries across 5 retrieval algorithms.",
        "summary": summary_df.to_dict(orient="records")
    }

@app.get("/api/benchmark/queries")
def get_benchmark_queries():
    return [
        {
            "query_id": q.query_id,
            "category": q.category,
            "query": q.query,
            "target_documents": q.target_documents,
            "ground_truth_relevant_count": q.ground_truth_relevant_count,
            "expected_keywords": q.expected_relevant_keywords,
            "required_focal_entities": q.required_focal_entities
        }
        for q in CLIMATE_BENCHMARK_QUERIES
    ]

# -----------------------------------------------------------------------------
# Static Files & Frontend Serving
# -----------------------------------------------------------------------------
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🌍 ClimateKG-RAG: Modern Web Application Server")
    print("Serving interactive UI at: http://localhost:8000")
    print("API Documentation at:      http://localhost:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
