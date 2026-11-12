import time
import logging
from typing import List, Dict, Any, Optional, Set
from config.settings import settings
from src.knowledge_graph.networkx_store import NetworkXGraphStore
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.normalizer import EntityNormalizer, SYNONYM_MAP
from src.retrieval.base import RetrievalResult, RetrievalResponse

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Knowledge Graph retrieval engine supporting multi-hop relational path discovery
    and evidence-linked traversal across NetworkX and Neo4j.
    """

    def __init__(
        self,
        nx_store: Optional[NetworkXGraphStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        max_hops: int = 2
    ):
        self.nx_store = nx_store or NetworkXGraphStore()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.max_hops = max_hops

    def identify_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Identifies focal entity nodes in the graph mentioned in the query."""
        q_low = query.lower()
        matched_nodes: Dict[str, Dict[str, Any]] = {}

        # 1. Check known synonym / canonical mapping
        for alias, (canon, etype) in SYNONYM_MAP.items():
            if alias in q_low:
                _, _, stable_id = EntityNormalizer.normalize(alias, etype)
                node = self.nx_store.get_node(stable_id)
                if node:
                    matched_nodes[stable_id] = node

        # 2. Check full graph node names
        direct_matches = self.nx_store.find_nodes_by_name(query)
        for node in direct_matches:
            matched_nodes[node["id"]] = node

        return list(matched_nodes.values())

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        """
        Executes multi-hop Knowledge Graph traversal for query entities.
        Returns ranked graph paths and evidence with exact document and page citations.
        """
        start_time = time.perf_counter()

        query_entities = self.identify_query_entities(query)
        if not query_entities:
            latency = (time.perf_counter() - start_time) * 1000
            return RetrievalResponse(
                query=query,
                method="Knowledge Graph",
                results=[],
                latency_ms=round(latency, 2),
                total_chunks_searched=self.nx_store.graph.number_of_edges()
            )

        candidate_paths: List[Dict[str, Any]] = []

        # Multi-hop traversal starting from each recognized focal entity
        for entity in query_entities:
            paths = self.nx_store.multi_hop_traversal(
                start_node_id=entity["id"],
                max_hops=self.max_hops,
                direction="both"
            )
            candidate_paths.extend(paths)

        # Score and rank candidate paths
        # Scoring incorporates:
        # - Inverse hop count (shorter paths preferred)
        # - Term overlap with query
        # - Confidence scores on relationships
        scored_results: List[RetrievalResult] = []
        seen_evidence_signatures: Set[str] = set()

        q_terms = set(query.lower().split())

        for path in candidate_paths:
            hops = path["hops"]
            nodes = path["nodes"]
            edges = path["edges"]

            # Construct readable path representation
            # e.g. [Carbon Dioxide] -(CAUSES)-> [Global Warming] -(AFFECTS)-> [Agricultural Yield Decline]
            path_str_parts = [f"[{nodes[0].get('name', 'Entity')}]"]
            evidence_snippets = []
            primary_edge = edges[0] if edges else {}

            for i, edge in enumerate(edges):
                rel_type = edge.get("type", "RELATED_TO")
                next_node_name = nodes[i + 1].get("name", "Entity") if i + 1 < len(nodes) else "Target"
                path_str_parts.append(f"-({rel_type})-> [{next_node_name}]")
                ev = edge.get("evidence_text", "").strip()
                if ev and ev not in evidence_snippets:
                    evidence_snippets.append(ev)

            path_description = " ".join(path_str_parts)
            evidence_text = "\n".join(evidence_snippets) if evidence_snippets else path_description

            # Canonical node set signature to eliminate reverse-direction duplicates (e.g. A->B vs B->A)
            node_ids_sorted = tuple(sorted(str(n.get("id", "")) for n in nodes))
            sig = (node_ids_sorted, hops)
            if sig in seen_evidence_signatures:
                continue
            seen_evidence_signatures.add(sig)

            # Calculate path score
            hop_penalty = 1.0 / (hops + 0.2)
            node_names_combined = " ".join(n.get("name", "").lower() for n in nodes)
            node_types_combined = " ".join(str(n.get("type", "")).lower() for n in nodes)
            all_path_text = f"{node_names_combined} {node_types_combined} {evidence_text.lower()} {path_description.lower()}"
            
            overlap_count = sum(1 for term in q_terms if len(term) > 3 and term in all_path_text)
            overlap_bonus = 0.20 * min(5, overlap_count)
            rel_confidence = sum(e.get("confidence", 0.8) for e in edges) / max(1, len(edges))

            # Query entity bridging bonus (paths connecting multiple entities mentioned in query)
            path_node_ids = set(str(n.get("id", "")).lower() for n in nodes)
            query_entity_ids = set(str(qe.get("id", "")).lower() for qe in query_entities)
            matched_query_ents = len(path_node_ids.intersection(query_entity_ids))
            relational_bonus = 0.25 * max(0, matched_query_ents - 1)

            final_score = round((0.3 * hop_penalty + 0.3 * rel_confidence + 0.2 * overlap_bonus + 0.2 * relational_bonus), 4)

            # Metadata provenance
            doc_id = primary_edge.get("doc_id", "doc_graph")
            filename = primary_edge.get("filename", "knowledge_graph")
            page_number = primary_edge.get("page_number", 1)
            chunk_id = primary_edge.get("source_chunk_id", f"graph_hop_{hops}")

            formatted_passage = f"GRAPH PATH: {path_description}\nEVIDENCE: {evidence_text}"

            scored_results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=filename.replace("_", " ").title(),
                filename=filename,
                page_number=page_number,
                text=formatted_passage,
                score=final_score,
                method="Knowledge Graph",
                metadata={
                    "path_description": path_description,
                    "hops": hops,
                    "node_ids": [n.get("id") for n in nodes],
                    "nodes": [n.get("name") for n in nodes],
                    "edges": [e.get("type") for e in edges],
                    "evidence_quotes": evidence_snippets
                }
            ))

        # Sort descending by score
        scored_results.sort(key=lambda r: r.score, reverse=True)

        for rank, res in enumerate(scored_results[:top_k], start=1):
            res.rank = rank

        latency = (time.perf_counter() - start_time) * 1000
        return RetrievalResponse(
            query=query,
            method="Knowledge Graph",
            results=scored_results[:top_k],
            latency_ms=round(latency, 2),
            total_chunks_searched=len(candidate_paths)
        )

