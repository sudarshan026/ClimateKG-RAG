import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import neo4j
from config.settings import settings
from src.knowledge_graph.schema import Entity, Relation, VALID_NODE_TYPES, VALID_RELATION_TYPES

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Neo4j Community Edition driver interface.
    Manages schemas, constraints, bulk ingestion, and multi-hop Cypher queries.
    Features read-only safeguards for user query execution.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE
        self._driver = None

    def connect(self) -> bool:
        """Attempts connection to Neo4j server."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=5
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j successfully at {self.uri}")
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection failed at {self.uri}: {e}")
            self._driver = None
            return False

    def is_connected(self) -> bool:
        return self._driver is not None

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def init_schema(self):
        """Creates unique constraints and indices for entity and document nodes."""
        if not self.is_connected():
            return

        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)"
        ]
        with self._driver.session(database=self.database) as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    logger.debug(f"Constraint notice: {e}")

    def add_entities_and_relations(self, entities: List[Entity], relations: List[Relation]):
        """Batch upserts entities and relations into Neo4j."""
        if not self.is_connected():
            return

        with self._driver.session(database=self.database) as session:
            # 1. Upsert entities with validated type
            for ent in entities:
                safe_type = ent.type if ent.type in VALID_NODE_TYPES else "ClimatePhenomenon"
                cypher = f"""
                MERGE (e:Entity {{id: $id}})
                SET e:{safe_type},
                    e.name = $name,
                    e.canonical_name = $canonical_name,
                    e.type = $type
                """
                session.run(cypher, id=ent.id, name=ent.name, canonical_name=ent.canonical_name, type=safe_type)

            # 2. Upsert relations with validated relation type
            for rel in relations:
                safe_rel_type = rel.type if rel.type in VALID_RELATION_TYPES else "RELATED_TO"
                cypher = f"""
                MATCH (s:Entity {{id: $source_id}})
                MATCH (t:Entity {{id: $target_id}})
                MERGE (s)-[r:{safe_rel_type} {{id: $id}}]->(t)
                SET r.evidence_text = $evidence,
                    r.source_chunk_id = $chunk_id,
                    r.doc_id = $doc_id,
                    r.filename = $filename,
                    r.page_number = $page,
                    r.confidence = $confidence,
                    r.is_inferred = $is_inferred
                """
                session.run(
                    cypher,
                    id=rel.id,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    evidence=rel.evidence_text,
                    chunk_id=rel.source_chunk_id,
                    doc_id=rel.doc_id,
                    filename=rel.filename,
                    page=rel.page_number,
                    confidence=rel.confidence,
                    is_inferred=rel.is_inferred
                )

    def remove_document(self, doc_id: str):
        """Removes all relations associated with doc_id and cleans up orphan entities."""
        if not self.is_connected():
            return
        try:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH ()-[r {doc_id: $doc_id}]->() DELETE r", doc_id=doc_id)
                session.run("MATCH (n:Entity) WHERE NOT (n)--() DELETE n")
        except Exception as e:
            logger.warning(f"Neo4j remove_document error: {e}")

    def execute_read_cypher(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a user or system Cypher query with strict read-only enforcement
        at both AST/regex level and database session transaction level.
        """
        if not self.is_connected():
            return []

        # Check for mutation keywords
        forbidden = re.compile(r'\b(CREATE|DELETE|DETACH|SET|REMOVE|DROP|MERGE)\b', re.IGNORECASE)
        if forbidden.search(cypher_query):
            raise ValueError("Read-only security violation: Mutating Cypher commands are forbidden.")

        with self._driver.session(database=self.database, default_access_mode=neo4j.AccessMode.READ) as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result]

    def multi_hop_traversal(self, entity_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Discovers multi-hop relational paths starting from a focal entity."""
        if not self.is_connected():
            return []

        cypher = f"""
        MATCH path = (start:Entity {{id: $entity_id}})-[r*1..{max_hops}]-(target:Entity)
        RETURN [n in nodes(path) | {{id: n.id, name: n.name, type: n.type}}] AS nodes,
               [rel in relationships(path) | {{
                   id: rel.id,
                   type: type(rel),
                   evidence: rel.evidence_text,
                   chunk_id: rel.source_chunk_id,
                   page: rel.page_number
               }}] AS relationships
        LIMIT 50
        """
        return self.execute_read_cypher(cypher, {"entity_id": entity_id})

    def get_statistics(self) -> Dict[str, Any]:
        """Queries Neo4j for node and relationship counts."""
        if not self.is_connected():
            return {"status": "offline", "nodes": 0, "edges": 0}

        try:
            with self._driver.session(database=self.database) as session:
                n_count = session.run("MATCH (n:Entity) RETURN count(n) AS cnt").single()["cnt"]
                r_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
                return {
                    "status": "connected",
                    "nodes": n_count,
                    "edges": r_count,
                    "uri": self.uri
                }
        except Exception as e:
            return {"status": "error", "message": str(e), "nodes": 0, "edges": 0}

    def clear(self):
        """Wipes graph data in Neo4j."""
        if not self.is_connected():
            return
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

