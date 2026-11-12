import pytest
import tempfile
from pathlib import Path
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.schema import Entity, Relation, NodeType, RelationType

def test_networkx_store_and_multihop():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = NetworkXGraphStore(storage_dir=Path(tmpdir))

        e1 = Entity(id="ent_co2", name="Carbon Dioxide", type=NodeType.GREENHOUSE_GAS.value, canonical_name="Carbon Dioxide")
        e2 = Entity(id="ent_gw", name="Global Warming", type=NodeType.CLIMATE_PHENOMENON.value, canonical_name="Global Warming")
        e3 = Entity(id="ent_yield", name="Agricultural Yield Decline", type=NodeType.CLIMATE_IMPACT.value, canonical_name="Agricultural Yield Decline")
        e4 = Entity(id="ent_drip", name="Drip Irrigation", type=NodeType.ADAPTATION_STRATEGY.value, canonical_name="Drip Irrigation")

        r1 = Relation(
            id="rel_1",
            source_id="ent_co2",
            source_name="Carbon Dioxide",
            target_id="ent_gw",
            target_name="Global Warming",
            type=RelationType.CAUSES.value,
            evidence_text="CO2 causes global warming.",
            source_chunk_id="chk_1",
            doc_id="doc_1",
            filename="report.txt",
            page_number=1
        )
        r2 = Relation(
            id="rel_2",
            source_id="ent_gw",
            source_name="Global Warming",
            target_id="ent_yield",
            target_name="Agricultural Yield Decline",
            type=RelationType.AFFECTS.value,
            evidence_text="Global warming affects crop yields.",
            source_chunk_id="chk_2",
            doc_id="doc_1",
            filename="report.txt",
            page_number=2
        )
        r3 = Relation(
            id="rel_3",
            source_id="ent_yield",
            source_name="Agricultural Yield Decline",
            target_id="ent_drip",
            target_name="Drip Irrigation",
            type=RelationType.ADAPTED_BY.value,
            evidence_text="Yield declines are adapted by drip irrigation.",
            source_chunk_id="chk_3",
            doc_id="doc_2",
            filename="adaptation.txt",
            page_number=1
        )

        store.batch_add([e1, e2, e3, e4], [r1, r2, r3])

        stats = store.get_statistics()
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 3

        # Test multi-hop traversal (2 hops from ent_co2)
        paths = store.multi_hop_traversal("ent_co2", max_hops=2)
        assert len(paths) >= 2
        # Check that 2-hop reaches ent_yield
        two_hop_nodes = [p["node_ids"][-1] for p in paths if p["hops"] == 2]
        assert "ent_yield" in two_hop_nodes

