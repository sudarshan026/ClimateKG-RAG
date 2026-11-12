import pytest
from src.chunking.chunker import DocumentChunk
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

@pytest.fixture
def sample_chunks():
    c1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Solar Energy Policy",
        filename="solar.txt",
        page_number=1,
        text="Solar energy and photovoltaics provide renewable zero-emission power.",
        word_count=8,
        char_start=0,
        char_end=60,
        chunk_index=0,
        metadata={"source": "solar.txt"}
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        doc_id="d2",
        doc_title="Indian Agriculture Drought",
        filename="agri.txt",
        page_number=1,
        text="Severe drought reduces wheat yields in northern India and triggers water scarcity.",
        word_count=11,
        char_start=0,
        char_end=80,
        chunk_index=0,
        metadata={"source": "agri.txt"}
    )
    return [c1, c2]

def test_tfidf_retriever(sample_chunks):
    retriever = TFIDFRetriever(sample_chunks)
    resp = retriever.retrieve("solar power renewable", top_k=2)
    assert resp.method == "TF-IDF"
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c1"
    assert resp.latency_ms >= 0

def test_bm25_retriever(sample_chunks):
    retriever = BM25Retriever(sample_chunks)
    resp = retriever.retrieve("drought and wheat in India", top_k=2)
    assert resp.method == "BM25"
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c2"

def test_dense_retriever(sample_chunks):
    retriever = DenseRetriever(sample_chunks)
    resp = retriever.retrieve("clean electricity photovoltaics", top_k=2)
    assert resp.method == "Dense Semantic"
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "c1"

