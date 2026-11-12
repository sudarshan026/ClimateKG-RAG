import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.document_processor.store import DocumentStore
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridGraphRAGRetriever
from src.qa.generator import GroundedAnswerGenerator
from src.evaluation.evaluator import ClimateEvaluator
from src.evaluation.benchmark_dataset import CLIMATE_BENCHMARK_QUERIES

def main():
    print("=" * 70)
    print("ClimateKG-RAG: Executing 32-Query IR & GraphRAG Benchmark")
    print("=" * 70)

    doc_store = DocumentStore()
    all_chunks = doc_store.get_all_chunks()
    if not all_chunks:
        print("[!] No documents indexed. Run scripts/ingest_corpus.py first.")
        return

    nx_store = NetworkXGraphStore()

    print("[+] Initializing Retrieval Engines...")
    tfidf = TFIDFRetriever(all_chunks)
    bm25 = BM25Retriever(all_chunks)
    dense = DenseRetriever(all_chunks)
    graph = GraphRetriever(nx_store=nx_store)
    hybrid = HybridGraphRAGRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        graph_retriever=graph,
        rrf_k=settings.RRF_K
    )
    generator = GroundedAnswerGenerator()

    evaluator = ClimateEvaluator(
        tfidf_retriever=tfidf,
        bm25_retriever=bm25,
        dense_retriever=dense,
        graph_retriever=graph,
        hybrid_retriever=hybrid,
        generator=generator
    )

    print(f"[+] Running evaluation over {len(CLIMATE_BENCHMARK_QUERIES)} benchmark queries...")
    results = evaluator.run_comparative_evaluation(top_k=settings.TOP_K_RETRIEVAL, compute_faithfulness=True)

    summary_df = results["summary_df"]
    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION SUMMARY TABLE (ACTUAL COMPUTED RESULTS)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Summary CSV exported to: {results['summary_csv']}")
    print(f"Detailed CSV exported to: {results['detailed_csv']}")

if __name__ == "__main__":
    main()

