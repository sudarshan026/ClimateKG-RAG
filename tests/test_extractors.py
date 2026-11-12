import pytest
from src.knowledge_graph.normalizer import EntityNormalizer, slugify
from src.knowledge_graph.schema import NodeType, RelationType
from src.knowledge_graph.extractor import KnowledgeGraphExtractor
from src.chunking.chunker import DocumentChunk

def test_slugify():
    assert slugify("Global Warming") == "global_warming"
    assert slugify("CO2 (Carbon Dioxide)") == "co2_carbon_dioxide"

def test_normalizer():
    canon, etype, s_id = EntityNormalizer.normalize("co2")
    assert canon == "Carbon Dioxide"
    assert etype == NodeType.GREENHOUSE_GAS.value
    assert s_id == "ent_greenhousegas_carbon_dioxide"

    canon2, etype2, s_id2 = EntityNormalizer.normalize("India")
    assert canon2 == "India"
    assert etype2 == NodeType.REGION.value
    assert s_id2 == "ent_region_india"

def test_deterministic_extractor():
    extractor = KnowledgeGraphExtractor()
    chunk_text = "Carbon dioxide causes global warming. In India, global warming causes agricultural yield decline."
    chunk = DocumentChunk(
        chunk_id="test_c1",
        doc_id="test_d1",
        doc_title="Test Doc",
        filename="test.txt",
        page_number=1,
        text=chunk_text,
        word_count=len(chunk_text.split()),
        char_start=0,
        char_end=len(chunk_text),
        chunk_index=0,
        metadata={"source": "test.txt"}
    )

    entities, relations = extractor.extract_deterministic(chunk)
    assert len(entities) >= 3
    ent_names = [e.canonical_name for e in entities]
    assert "Carbon Dioxide" in ent_names
    assert "Global Warming" in ent_names
    assert "India" in ent_names

    assert len(relations) >= 1
    rel_types = [r.type for r in relations]
    assert RelationType.CAUSES.value in rel_types
    for r in relations:
        assert r.source_chunk_id == "test_c1"
        assert len(r.evidence_text) > 0

