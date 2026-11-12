import time
import logging
from typing import List, Dict, Any, Optional

from config.settings import settings
from src.chunking.chunker import DocumentChunk
from src.retrieval.base import RetrievalResult, RetrievalResponse
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranking import ReciprocalRankFusion

logger = logging.getLogger(__name__)

class HybridGraphRAGRetriever:
    """
    Unified Hybrid GraphRAG Retrieval Engine.
    Executes Lexical (BM25), Semantic (Dense), and Structural (Knowledge Graph)
    retrieval pipelines, performing Reciprocal Rank Fusion (RRF) reranking.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        graph_retriever: GraphRetriever,
        rrf_k: int = 60
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.graph_retriever = graph_retriever
        self.fuser = ReciprocalRankFusion(k=rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> RetrievalResponse:
        """
        Executes hybrid retrieval and fuses results using RRF.
        Default weights: BM25 (1.0), Dense (1.2), Knowledge Graph (1.5).
        """
        start_time = time.perf_counter()

        default_weights = {
            "BM25": 1.0,
            "Dense Semantic": 1.2,
            "Knowledge Graph": 1.5
        }
        active_weights = weights or default_weights

        # 1. Execute individual retrievals
        bm25_resp = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        dense_resp = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        graph_resp = self.graph_retriever.retrieve(query, top_k=top_k * 2)

        ranked_lists = {
            "BM25": bm25_resp.results,
            "Dense Semantic": dense_resp.results,
            "Knowledge Graph": graph_resp.results
        }

        # 2. Rerank via Reciprocal Rank Fusion
        fused_results = self.fuser.fuse(
            ranked_lists=ranked_lists,
            weights=active_weights,
            top_n=top_k
        )

        total_searched = (
            bm25_resp.total_chunks_searched +
            dense_resp.total_chunks_searched +
            graph_resp.total_chunks_searched
        )

        latency = (time.perf_counter() - start_time) * 1000

        # Attach sub-pipeline latencies in metadata of top response
        return RetrievalResponse(
            query=query,
            method="Hybrid GraphRAG",
            results=fused_results,
            latency_ms=round(latency, 2),
            total_chunks_searched=total_searched
        )

