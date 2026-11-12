import pytest
from src.chunking.chunker import DocumentChunk
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.schema import Entity, Relation, NodeType, RelationType
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridGraphRAGRetriever
from src.retrieval.reranking import ReciprocalRankFusion
from src.qa.generator import GroundedAnswerGenerator

def test_rrf_fusion():
    fuser = ReciprocalRankFusion(k=60)
    from src.retrieval.base import RetrievalResult
    r1 = RetrievalResult(chunk_id="c1", doc_id="d1", doc_title="D1", filename="f1.txt", page_number=1, text="Text 1", score=0.9, rank=1, method="M1")
    r2 = RetrievalResult(chunk_id="c2", doc_id="d2", doc_title="D2", filename="f2.txt", page_number=1, text="Text 2", score=0.8, rank=2, method="M1")
    r3 = RetrievalResult(chunk_id="c1", doc_id="d1", doc_title="D1", filename="f1.txt", page_number=1, text="Text 1", score=0.95, rank=1, method="M2")

    fused = fuser.fuse(
        ranked_lists={"M1": [r1, r2], "M2": [r3]},
        weights={"M1": 1.0, "M2": 1.0},
        top_n=2
    )

    assert len(fused) == 2
    # c1 appeared at rank 1 in both lists, so it must be top ranked
    assert fused[0].chunk_id == "c1"
    assert fused[0].score > fused[1].score

def test_grounded_answer_generator_insufficient():
    generator = GroundedAnswerGenerator()
    resp = generator.answer_query(
        query="What is the GDP of Mars in 2050?",
        retrieval_method="Hybrid GraphRAG",
        results=[]
    )
    assert resp.confidence == "Insufficient Evidence"
    assert "Insufficient Evidence" in resp.answer

