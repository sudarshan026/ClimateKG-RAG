"""
End-to-end integration tests for ClimateKG-RAG.

Validates the full pipeline:
1. Document ingestion and chunking
2. Knowledge Graph construction and schema adherence
3. Document deletion and graph consistency (orphan pruning)
4. All 5 retrieval engines (TF-IDF, BM25, Dense, Graph, Hybrid GraphRAG)
5. Reciprocal Rank Fusion (RRF) and deduplication
6. Evidence-grounded answer generation and insufficient evidence handling
7. Evaluation metric mathematical invariants (MAP <= 1.0, claim evidence support)
8. Graph serialization and persistence
"""

import os
import tempfile
import pytest
from pathlib import Path

from src.document_processor.cleaner import clean_text
from src.chunking.chunker import TextChunker, DocumentChunk
from src.document_processor.extractor import ProcessedDocument, PageData, DocumentExtractor, compute_file_hash
from src.document_processor.store import DocumentStore
from src.knowledge_graph.schema import Entity, Relation, NodeType, RelationType
from src.knowledge_graph.normalizer import EntityNormalizer
from src.knowledge_graph.extractor import KnowledgeGraphExtractor
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.graph_builder import GraphBuilder
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridGraphRAGRetriever
from src.retrieval.reranking import ReciprocalRankFusion
from src.qa.generator import GroundedAnswerGenerator
from src.evaluation.metrics import RetrievalMetrics


@pytest.fixture
def sample_chunks():
    """Provides a deterministic set of realistic climate document chunks."""
    c1 = DocumentChunk(
        chunk_id="chk_ipcc_001",
        doc_id="doc_ipcc",
        doc_title="IPCC AR6 Working Group I SPM",
        filename="ipcc_ar6_spm.txt",
        page_number=1,
        text="Global warming caused by greenhouse gas emissions drives severe droughts and extreme heatwaves in India. These climate impacts directly affect agriculture and food security.",
        word_count=23,
        char_start=0,
        char_end=173,
        chunk_index=0,
        metadata={"source": "ipcc_ar6_spm.txt"}
    )
    c2 = DocumentChunk(
        chunk_id="chk_icar_001",
        doc_id="doc_icar",
        doc_title="ICAR NICRA Adaptation Strategies",
        filename="icar_nicra.txt",
        page_number=1,
        text="Indian agriculture is adapted by drought-tolerant crop varieties and drip irrigation systems promoted by ICAR under the National Innovations in Climate Resilient Agriculture.",
        word_count=24,
        char_start=0,
        char_end=177,
        chunk_index=0,
        metadata={"source": "icar_nicra.txt"}
    )
    c3 = DocumentChunk(
        chunk_id="chk_unep_001",
        doc_id="doc_unep",
        doc_title="UNEP Emissions Gap Report 2023",
        filename="unep_egr_2023.txt",
        page_number=1,
        text="Global carbon dioxide emissions reached record levels. Renewable energy expansion and carbon pricing policies are critical mitigation strategies to limit temperature rise to 1.5C.",
        word_count=26,
        char_start=0,
        char_end=185,
        chunk_index=0,
        metadata={"source": "unep_egr_2023.txt"}
    )
    return [c1, c2, c3]


# ---------------------------------------------------------------------------
# 1. Document Processor & Chunking Tests
# ---------------------------------------------------------------------------
def test_text_cleaner():
    raw = "  Global   warming\r\n\r\nincreases   extreme heat.  \x0c"
    cleaned = clean_text(raw)
    assert "Global warming" in cleaned
    assert "extreme heat." in cleaned


def test_chunker_sliding_window():
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    text = "Sentence one about climate change. Sentence two about global warming. Sentence three about droughts in India. Sentence four about agriculture."
    doc = ProcessedDocument(
        doc_id="doc1",
        filename="doc1.txt",
        title="Doc Title",
        file_type="txt",
        file_hash="hash1",
        total_pages=1,
        pages=[{"page_number": 1, "text": text, "cleaned_text": text}],
        full_text=text
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chk in chunks:
        assert chk.doc_id == "doc1"
        assert chk.chunk_id.startswith("doc1_p1_c")


def test_document_store_duplicate_detection(tmp_path):
    store = DocumentStore(storage_dir=tmp_path / "doc_store")
    raw_text = "This is a climate document for duplicate verification testing."
    
    doc1 = ProcessedDocument(
        doc_id="doc_test_1",
        filename="test_doc.txt",
        title="Test Document",
        file_type="txt",
        file_hash="hash_identical_123",
        total_pages=1,
        pages=[{"page_number": 1, "text": raw_text, "cleaned_text": raw_text}],
        full_text=raw_text
    )
    chk1 = DocumentChunk(
        chunk_id="chk_test_1",
        doc_id="doc_test_1",
        doc_title="Test Document",
        filename="test_doc.txt",
        page_number=1,
        text=raw_text,
        word_count=8,
        char_start=0,
        char_end=len(raw_text),
        chunk_index=0,
        metadata={}
    )
    
    added1 = store.add_document(doc1, [chk1])
    assert added1 is True
    assert store.get_statistics()["total_documents"] == 1

    # Second ingestion of identical content hash should be detected as duplicate
    doc2 = ProcessedDocument(
        doc_id="doc_test_2",
        filename="test_doc_dup.txt",
        title="Test Document Duplicate",
        file_type="txt",
        file_hash="hash_identical_123",  # Identical hash
        total_pages=1,
        pages=[{"page_number": 1, "text": raw_text, "cleaned_text": raw_text}],
        full_text=raw_text
    )
    assert store.is_duplicate(doc2.file_hash) == "doc_test_1"


# ---------------------------------------------------------------------------
# 2. Knowledge Graph Extraction, Normalization & Schema Tests
# ---------------------------------------------------------------------------
def test_entity_normalization():
    canon, etype, sid = EntityNormalizer.normalize("co2")
    assert canon == "Carbon Dioxide"
    assert etype == NodeType.GREENHOUSE_GAS.value
    assert sid == "ent_greenhousegas_carbon_dioxide"

    canon2, etype2, sid2 = EntityNormalizer.normalize("India")
    assert canon2 == "India"
    assert etype2 == NodeType.REGION.value
    assert sid2 == "ent_region_india"


def test_rule_based_extraction_and_graph_population(tmp_path):
    nx_store = NetworkXGraphStore(storage_dir=tmp_path / "test_graph")
    extractor = KnowledgeGraphExtractor()

    chunk_text = "Global warming causes extreme droughts in India. Indian agriculture is adapted by drip irrigation."
    chunk = DocumentChunk(
        chunk_id="chk_test_1",
        doc_id="doc_test",
        doc_title="Test Doc",
        filename="test.txt",
        page_number=1,
        text=chunk_text,
        word_count=len(chunk_text.split()),
        char_start=0,
        char_end=len(chunk_text),
        chunk_index=0,
        metadata={}
    )
    entities, relations = extractor.extract_deterministic(chunk)

    assert len(entities) >= 3
    assert len(relations) >= 1

    for ent in entities:
        nx_store.add_entity(ent)
    for rel in relations:
        nx_store.add_relation(rel)

    stats = nx_store.get_statistics()
    assert stats["total_nodes"] >= 3
    assert stats["total_edges"] >= 1


# ---------------------------------------------------------------------------
# 3. Document Deletion & Graph Consistency (Edge & Orphan Pruning)
# ---------------------------------------------------------------------------
def test_graph_document_deletion_consistency(tmp_path):
    nx_store = NetworkXGraphStore(storage_dir=tmp_path / "test_deletion")
    builder = GraphBuilder(networkx_store=nx_store, neo4j_client=None)

    # Document A creates Node A1, Node A2, Edge A1 -> A2
    e_a1 = Entity(id="ent_a1", name="NodeA1", type=NodeType.CLIMATE_PHENOMENON.value, canonical_name="NodeA1")
    e_a2 = Entity(id="ent_a2", name="NodeA2", type=NodeType.CLIMATE_IMPACT.value, canonical_name="NodeA2")
    r_a = Relation(
        id="rel_a",
        source_id="ent_a1",
        source_name="NodeA1",
        target_id="ent_a2",
        target_name="NodeA2",
        type=RelationType.CAUSES.value,
        evidence_text="A1 causes A2.",
        source_chunk_id="chk_a",
        doc_id="doc_a",
        filename="doc_a.txt",
        page_number=1
    )

    # Document B creates Node B1, Node A2 (shared), Edge B1 -> A2
    e_b1 = Entity(id="ent_b1", name="NodeB1", type=NodeType.GREENHOUSE_GAS.value, canonical_name="NodeB1")
    r_b = Relation(
        id="rel_b",
        source_id="ent_b1",
        source_name="NodeB1",
        target_id="ent_a2",
        target_name="NodeA2",
        type=RelationType.CAUSES.value,
        evidence_text="B1 causes A2.",
        source_chunk_id="chk_b",
        doc_id="doc_b",
        filename="doc_b.txt",
        page_number=1
    )

    nx_store.add_entity(e_a1)
    nx_store.add_entity(e_a2)
    nx_store.add_entity(e_b1)
    nx_store.add_relation(r_a)
    nx_store.add_relation(r_b)
    nx_store._save()

    assert nx_store.graph.number_of_nodes() == 3
    assert nx_store.graph.number_of_edges() == 2

    # Remove Document A
    removed_edges = builder.remove_document("doc_a")
    assert removed_edges == 1

    # Node B1 and Node A2 should still exist because Document B links them
    assert nx_store.graph.has_node("ent_b1")
    assert nx_store.graph.has_node("ent_a2")
    assert not nx_store.graph.has_node("ent_a1")  # Orphan node A1 was pruned
    assert nx_store.graph.number_of_edges() == 1


# ---------------------------------------------------------------------------
# 4. Multi-Method Retrieval & RRF Fusion Tests
# ---------------------------------------------------------------------------
def test_all_five_retrieval_engines(sample_chunks, tmp_path):
    # Setup Graph
    nx_store = NetworkXGraphStore(storage_dir=tmp_path / "test_retrieval_graph")
    extractor = KnowledgeGraphExtractor()
    for chk in sample_chunks:
        ents, rels = extractor.extract_deterministic(chk)
        for e in ents:
            nx_store.add_entity(e)
        for r in rels:
            nx_store.add_relation(r)

    # Initialize all 5 retrievers
    tfidf = TFIDFRetriever(sample_chunks)
    bm25 = BM25Retriever(sample_chunks)
    dense = DenseRetriever(sample_chunks)
    graph = GraphRetriever(nx_store=nx_store)
    hybrid = HybridGraphRAGRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        graph_retriever=graph,
        rrf_k=60
    )

    query = "How does global warming affect agriculture in India and what are the adaptation strategies?"

    # 1. TF-IDF
    tfidf_resp = tfidf.retrieve(query, top_k=2)
    assert tfidf_resp.method == "TF-IDF"
    assert len(tfidf_resp.results) > 0
    assert tfidf_resp.results[0].score >= 0.0

    # 2. BM25
    bm25_resp = bm25.retrieve(query, top_k=2)
    assert bm25_resp.method == "BM25"
    assert len(bm25_resp.results) > 0

    # 3. Dense
    dense_resp = dense.retrieve(query, top_k=2)
    assert dense_resp.method == "Dense Semantic"
    assert len(dense_resp.results) > 0

    # 4. Graph
    graph_resp = graph.retrieve(query, top_k=5)
    assert graph_resp.method == "Knowledge Graph"
    assert isinstance(graph_resp.results, list)

    # 5. Hybrid GraphRAG
    hybrid_resp = hybrid.retrieve(query, top_k=2)
    assert hybrid_resp.method == "Hybrid GraphRAG"
    assert len(hybrid_resp.results) > 0
    # Deduplication check: every chunk_id in hybrid_resp should be unique
    chunk_ids = [r.chunk_id for r in hybrid_resp.results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_deduplication():
    fuser = ReciprocalRankFusion(k=60)
    from src.retrieval.base import RetrievalResult
    # Two identical chunk_ids inside the same method list shouldn't happen, but fuser should handle duplicates cleanly
    r1 = RetrievalResult(chunk_id="c1", doc_id="d1", doc_title="D1", filename="f1.txt", page_number=1, text="T1", score=0.9, rank=1, method="M1")
    r2 = RetrievalResult(chunk_id="c1", doc_id="d1", doc_title="D1", filename="f1.txt", page_number=1, text="T1", score=0.8, rank=2, method="M1")
    r3 = RetrievalResult(chunk_id="c2", doc_id="d2", doc_title="D2", filename="f2.txt", page_number=1, text="T2", score=0.7, rank=1, method="M2")

    fused = fuser.fuse(
        ranked_lists={"M1": [r1, r2], "M2": [r3]},
        weights={"M1": 1.0, "M2": 1.0},
        top_n=5
    )
    # Output must have distinct chunk IDs
    fused_ids = [r.chunk_id for r in fused]
    assert len(fused_ids) == len(set(fused_ids))
    assert fused_ids == ["c1", "c2"]


# ---------------------------------------------------------------------------
# 5. Grounded QA & Evidence Handling Tests
# ---------------------------------------------------------------------------
def test_grounded_answer_with_valid_evidence(sample_chunks):
    generator = GroundedAnswerGenerator()
    from src.retrieval.base import RetrievalResult
    r = RetrievalResult(
        chunk_id=sample_chunks[0].chunk_id,
        doc_id=sample_chunks[0].doc_id,
        doc_title=sample_chunks[0].doc_title,
        filename=sample_chunks[0].filename,
        page_number=sample_chunks[0].page_number,
        text=sample_chunks[0].text,
        score=0.95,
        rank=1,
        method="BM25"
    )
    ans = generator.answer_query("What does global warming cause in India?", "BM25", [r])
    assert ans.confidence in ["High", "Medium"]
    assert len(ans.citations) > 0
    assert ans.citations[0].doc_title == "IPCC AR6 Working Group I SPM"
    assert "droughts" in ans.answer.lower() or "heatwaves" in ans.answer.lower()


def test_insufficient_evidence_response():
    generator = GroundedAnswerGenerator()
    ans = generator.answer_query("What is the extraterrestrial atmospheric pressure of Venus?", "Hybrid GraphRAG", [])
    assert ans.confidence == "Insufficient Evidence"
    assert "Insufficient Evidence" in ans.answer
    assert len(ans.citations) == 0


# ---------------------------------------------------------------------------
# 6. Evaluation Metric Invariant Tests
# ---------------------------------------------------------------------------
def test_average_precision_upper_bound():
    """Verify that AP and MAP never exceed 1.0 under any retrieval condition."""
    # Scenario A: Perfect retrieval (3 relevant retrieved out of 3 total)
    ap_perfect = RetrievalMetrics.average_precision([1, 1, 1], total_relevant=3, k=3)
    assert ap_perfect == 1.0

    # Scenario B: Over-retrieval (more relevant items retrieved than gold total due to duplicate count)
    ap_over = RetrievalMetrics.average_precision([1, 1, 1, 1], total_relevant=2, k=4)
    assert ap_over <= 1.0

    # Scenario C: Zero relevant retrieved
    ap_zero = RetrievalMetrics.average_precision([0, 0, 0], total_relevant=3, k=3)
    assert ap_zero == 0.0

    # Scenario D: Mixed ranks
    # Ranks: 2 (P=1/2=0.5), 4 (P=2/4=0.5). AP = (0.5 + 0.5) / 2 = 0.5
    ap_mixed = RetrievalMetrics.average_precision([0, 1, 0, 1, 0], total_relevant=2, k=5)
    assert ap_mixed == 0.5
    assert 0.0 <= ap_mixed <= 1.0


def test_claim_evidence_support_metric():
    evidence = [
        "Renewable energy expansion and carbon pricing policies are critical mitigation strategies to limit global warming."
    ]
    # Grounded answer:
    grounded_ans = "Renewable energy and carbon pricing are critical mitigation strategies to limit warming."
    score_high = RetrievalMetrics.calculate_claim_evidence_support(grounded_ans, evidence)
    assert score_high >= 0.7

    # Fabricated / ungrounded answer:
    hallucinated_ans = "The government announced a 50% discount on interplanetary tickets to Jupiter."
    score_low = RetrievalMetrics.calculate_claim_evidence_support(hallucinated_ans, evidence)
    assert score_low <= 0.3


# ---------------------------------------------------------------------------
# 7. Graph Serialization & Persistence Tests
# ---------------------------------------------------------------------------
def test_networkx_graph_persistence(tmp_path):
    storage_dir = tmp_path / "persistent_graph"
    store1 = NetworkXGraphStore(storage_dir=storage_dir)

    ent = Entity(
        id="ent_greenhousegas_carbon_dioxide",
        name="Carbon Dioxide",
        type=NodeType.GREENHOUSE_GAS.value,
        canonical_name="Carbon Dioxide"
    )
    store1.add_entity(ent)
    store1._save()

    # Load in new instance from disk
    store2 = NetworkXGraphStore(storage_dir=storage_dir)
    assert store2.graph.has_node("ent_greenhousegas_carbon_dioxide")
    loaded_ent = store2.get_node("ent_greenhousegas_carbon_dioxide")
    assert loaded_ent["canonical_name"] == "Carbon Dioxide"
    assert loaded_ent["type"] == NodeType.GREENHOUSE_GAS.value
