import logging
from typing import List, Dict, Any, Optional, Tuple
from config.settings import settings
from src.chunking.chunker import DocumentChunk
from src.knowledge_graph.schema import Entity, Relation
from src.knowledge_graph.extractor import KnowledgeGraphExtractor
from src.knowledge_graph.normalizer import EntityNormalizer
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Coordinates Knowledge Graph extraction, normalization, and persistence
    across NetworkX storage and Neo4j graph database.
    """

    def __init__(
        self,
        networkx_store: Optional[NetworkXGraphStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        extractor: Optional[KnowledgeGraphExtractor] = None
    ):
        self.nx_store = networkx_store or NetworkXGraphStore()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.extractor = extractor or KnowledgeGraphExtractor()

        # Attempt Neo4j connection if enabled
        if settings.USE_NEO4J_IF_AVAILABLE:
            connected = self.neo4j_client.connect()
            if connected:
                self.neo4j_client.init_schema()

    def process_chunks(self, chunks: List[DocumentChunk], progress_callback=None) -> Dict[str, Any]:
        """
        Extracts entities and relationships from chunks, normalizes,
        and persists them to graph backends.
        """
        total_chunks = len(chunks)
        all_entities: Dict[str, Entity] = {}
        all_relations: List[Relation] = []

        for idx, chunk in enumerate(chunks):
            ents, rels = self.extractor.extract(chunk)

            for e in ents:
                all_entities[e.id] = e

            all_relations.extend(rels)

            if progress_callback:
                progress_callback(idx + 1, total_chunks)

        entity_list = list(all_entities.values())

        # Persist to NetworkX store
        self.nx_store.batch_add(entity_list, all_relations)

        # Persist to Neo4j if available
        neo4j_synced = False
        if self.neo4j_client.is_connected():
            try:
                self.neo4j_client.add_entities_and_relations(entity_list, all_relations)
                neo4j_synced = True
            except Exception as e:
                logger.warning(f"Neo4j sync failed: {e}")

        return {
            "chunks_processed": total_chunks,
            "entities_extracted": len(entity_list),
            "relations_extracted": len(all_relations),
            "neo4j_synced": neo4j_synced
        }

    def remove_document(self, doc_id: str) -> int:
        """
        Removes all graph edges and cleans up orphan entities associated with doc_id
        across both NetworkX and Neo4j backends.
        """
        edges_removed = self.nx_store.remove_document(doc_id)
        if self.neo4j_client.is_connected():
            self.neo4j_client.remove_document(doc_id)
        return edges_removed

    def clear(self):
        """Wipes knowledge graph across backends."""
        self.nx_store.clear()
        if self.neo4j_client.is_connected():
            self.neo4j_client.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics from the primary graph store and Neo4j connectivity status."""
        stats = self.nx_store.get_statistics()
        neo4j_status = "Connected" if self.neo4j_client.is_connected() else "Offline / Standalone mode"
        stats["neo4j_status"] = neo4j_status
        return stats

