import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from data.sample_climate_docs import generate_sample_corpus
from src.document_processor.extractor import DocumentExtractor
from src.document_processor.store import DocumentStore
from src.chunking.chunker import TextChunker
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

def run_ingestion(force_reindex: bool = False):
    print("=" * 65)
    print("ClimateKG-RAG: Corpus Ingestion and Knowledge Graph Construction")
    print("=" * 65)

    settings.ensure_directories()
    doc_store = DocumentStore()
    chunker = TextChunker(chunk_size=settings.DEFAULT_CHUNK_SIZE, chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP)
    nx_store = NetworkXGraphStore()
    graph_builder = GraphBuilder(networkx_store=nx_store)

    if force_reindex:
        print("[*] Clearing existing indices, cache, and graph store...")
        doc_store.clear()
        graph_builder.clear()
        # Clean up old raw files to ensure authentic corpus
        for old_f in settings.RAW_DATA_DIR.glob("*.*"):
            old_f.unlink(missing_ok=True)
        # Clear vector cache
        emb_file = settings.VECTOR_CACHE_DIR / "embeddings.npz"
        emb_file.unlink(missing_ok=True)

    # Step 1: Ensure sample documents exist in data/raw
    raw_files = list(settings.RAW_DATA_DIR.glob("*.*"))
    if not raw_files or force_reindex:
        print("[+] Generating authentic climate research corpus...")
        generated = generate_sample_corpus()
        print(f"    Generated {len(generated)} authentic climate science texts.")
        raw_files = list(settings.RAW_DATA_DIR.glob("*.*"))

    # Step 2: Extract, chunk, and index documents
    print(f"\n[+] Ingesting {len(raw_files)} documents from {settings.RAW_DATA_DIR}...")
    new_chunks = []
    for fpath in raw_files:
        if fpath.suffix.lower() not in settings.SUPPORTED_EXTENSIONS:
            continue
        try:
            doc = DocumentExtractor.extract(fpath, fpath.name)
            chunks = chunker.chunk_document(doc)
            added = doc_store.add_document(doc, chunks)
            if added:
                new_chunks.extend(chunks)
                print(f"    [OK] Ingested: {fpath.name} ({len(chunks)} chunks, {doc.total_pages} pages)")
            else:
                print(f"    [-] Already indexed: {fpath.name}")
        except Exception as e:
            print(f"    [ERR] Failed to process {fpath.name}: {e}")

    all_chunks = doc_store.get_all_chunks()
    print(f"\n[+] Total chunks in DocumentStore: {len(all_chunks)}")

    # Step 3: Knowledge Graph Extraction
    print("\n[+] Constructing Climate Knowledge Graph from chunks...")
    kg_result = graph_builder.process_chunks(all_chunks)
    print(f"    Chunks analyzed:      {kg_result['chunks_processed']}")
    print(f"    Entities extracted:   {kg_result['entities_extracted']}")
    print(f"    Relations extracted:  {kg_result['relations_extracted']}")
    print(f"    Neo4j synchronized:   {kg_result['neo4j_synced']}")

    # Step 4: Index Retrieval Models
    print("\n[+] Pre-indexing Retrieval Engines...")
    tfidf = TFIDFRetriever(all_chunks)
    print("    [OK] TF-IDF Index built.")

    bm25 = BM25Retriever(all_chunks)
    print("    [OK] BM25 Index built.")

    print("    [*] Computing Dense Semantic Embeddings (Sentence Transformers)...")
    dense = DenseRetriever(all_chunks)
    print("    [OK] Dense Semantic Embeddings cached to disk.")

    # Step 5: Computed Live Statistics
    stats = doc_store.get_statistics()
    g_stats = graph_builder.get_statistics()

    print("\n" + "=" * 65)
    print("INGESTION & GRAPH CONSTRUCTION COMPLETE")
    print("=" * 65)
    print(f"Total Documents:           {stats['total_documents']}")
    print(f"Total Chunks:              {stats['total_chunks']}")
    print(f"Total Words:               {stats['total_words']}")
    print(f"Graph Nodes (Entities):    {g_stats['total_nodes']}")
    print(f"Graph Edges (Relations):   {g_stats['total_edges']}")
    print(f"Graph Density:             {g_stats['density']}")
    print(f"Neo4j Status:              {g_stats['neo4j_status']}")
    print("=" * 65)

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_ingestion(force_reindex=force)

