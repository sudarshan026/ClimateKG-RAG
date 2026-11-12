import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

from config.settings import settings
from src.knowledge_graph.schema import Entity, Relation, NodeType, RelationType

logger = logging.getLogger(__name__)

class NetworkXGraphStore:
    """
    In-memory and file-persistent Knowledge Graph storage using NetworkX MultiDiGraph.
    Provides graph persistence, multi-hop path search, neighborhood extraction,
    and topological statistics.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.GRAPH_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_dir / "graph.json"
        self.graph = nx.MultiDiGraph()
        self._load()

    def _save(self):
        """Serializes NetworkX graph to JSON."""
        data = {
            "nodes": [
                {
                    "id": node_id,
                    **self.graph.nodes[node_id]
                }
                for node_id in self.graph.nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "key": key,
                    **data
                }
                for u, v, key, data in self.graph.edges(keys=True, data=True)
            ]
        }
        try:
            with open(self.graph_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to persist graph to disk: {e}")

    def _load(self):
        """Deserializes graph from JSON if file exists."""
        if not self.graph_file.exists():
            return
        try:
            with open(self.graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.graph.clear()
            for n in data.get("nodes", []):
                nid = n["id"]
                attrs = {k: v for k, v in n.items() if k != "id"}
                self.graph.add_node(nid, **attrs)

            for e in data.get("edges", []):
                u = e["source"]
                v = e["target"]
                key = e.get("key", 0)
                attrs = {k: v for k, v in e.items() if k not in ("source", "target", "key")}
                self.graph.add_edge(u, v, key=key, **attrs)

            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"Could not load existing graph from disk: {e}")
            self.graph.clear()

    def add_entity(self, entity: Entity):
        """Adds or updates an entity node."""
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            canonical_name=entity.canonical_name,
            description=entity.description,
            properties=entity.properties
        )

    def add_relation(self, relation: Relation):
        """Adds an evidence-linked relationship edge."""
        # Ensure source and target exist
        if not self.graph.has_node(relation.source_id):
            self.graph.add_node(
                relation.source_id,
                name=relation.source_name,
                type=NodeType.CLIMATE_PHENOMENON.value,
                canonical_name=relation.source_name
            )
        if not self.graph.has_node(relation.target_id):
            self.graph.add_node(
                relation.target_id,
                name=relation.target_name,
                type=NodeType.CLIMATE_PHENOMENON.value,
                canonical_name=relation.target_name
            )

        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            relation_id=relation.id,
            type=relation.type,
            evidence_text=relation.evidence_text,
            source_chunk_id=relation.source_chunk_id,
            doc_id=relation.doc_id,
            filename=relation.filename,
            page_number=relation.page_number,
            confidence=relation.confidence,
            is_inferred=relation.is_inferred,
            properties=relation.properties
        )

    def batch_add(self, entities: List[Entity], relations: List[Relation]):
        """Adds multiple entities and relations, then persists."""
        for ent in entities:
            self.add_entity(ent)
        for rel in relations:
            self.add_relation(rel)
        self._save()

    def remove_document(self, doc_id: str) -> int:
        """
        Removes all relationship edges associated with doc_id.
        Cleans up any orphaned entity nodes that have no remaining edges.
        Returns the number of edges removed.
        """
        edges_to_remove = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get("doc_id") == doc_id:
                edges_to_remove.append((u, v, key))

        for u, v, key in edges_to_remove:
            self.graph.remove_edge(u, v, key=key)

        # Clean up orphan nodes with degree 0
        orphan_nodes = [n for n in list(self.graph.nodes) if self.graph.degree(n) == 0]
        for n in orphan_nodes:
            self.graph.remove_node(n)

        self._save()
        logger.info(f"Removed {len(edges_to_remove)} edges and {len(orphan_nodes)} orphan nodes for doc_id {doc_id}")
        return len(edges_to_remove)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(node_id):
            return {"id": node_id, **self.graph.nodes[node_id]}
        return None

    def find_nodes_by_name(self, query: str) -> List[Dict[str, Any]]:
        """Finds entities matching query string in name or canonical_name."""
        q_low = query.lower().strip()
        matched = []
        for nid, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            canon = data.get("canonical_name", "").lower()
            if q_low in name or q_low in canon or name in q_low:
                matched.append({"id": nid, **data})
        return matched

    def multi_hop_traversal(
        self,
        start_node_id: str,
        max_hops: int = 2,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Explores paths up to max_hops from start_node_id.
        Returns detailed paths with node metadata, edge relations, and exact evidence text.
        """
        if not self.graph.has_node(start_node_id):
            return []

        paths_found: List[Dict[str, Any]] = []
        visited_nodes = {start_node_id}

        # Convert to undirected view if direction is both
        work_graph = self.graph.to_undirected(as_view=True) if direction == "both" else self.graph

        def dfs(current_node: str, current_path_nodes: List[str], current_edges: List[Dict[str, Any]], current_hop: int):
            if current_hop > max_hops:
                return

            if current_hop > 0:
                paths_found.append({
                    "hops": current_hop,
                    "node_ids": list(current_path_nodes),
                    "nodes": [{"id": nid, **self.graph.nodes[nid]} for nid in current_path_nodes],
                    "edges": list(current_edges)
                })

            for neighbor in work_graph.neighbors(current_node):
                if neighbor in current_path_nodes:
                    continue  # Prevent cycles

                # Retrieve edge attributes from original directed graph
                edge_data_list = []
                if self.graph.has_edge(current_node, neighbor):
                    for k, d in self.graph.get_edge_data(current_node, neighbor).items():
                        edge_data_list.append({"from": current_node, "to": neighbor, **d})
                elif self.graph.has_edge(neighbor, current_node):
                    for k, d in self.graph.get_edge_data(neighbor, current_node).items():
                        edge_data_list.append({"from": neighbor, "to": current_node, **d})

                for edata in edge_data_list:
                    dfs(
                        neighbor,
                        current_path_nodes + [neighbor],
                        current_edges + [edata],
                        current_hop + 1
                    )

        dfs(start_node_id, [start_node_id], [], 0)
        return paths_found

    def get_subgraph_for_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Extracts subgraph induced by the given node IDs plus their immediate 1-hop edges."""
        sub_nodes = set(node_ids)
        # Include connected neighbors
        for nid in node_ids:
            if self.graph.has_node(nid):
                for succ in self.graph.successors(nid):
                    sub_nodes.add(succ)
                for pred in self.graph.predecessors(nid):
                    sub_nodes.add(pred)

        nodes_list = []
        for nid in sub_nodes:
            if self.graph.has_node(nid):
                nodes_list.append({"id": nid, **self.graph.nodes[nid]})

        edges_list = []
        for u in sub_nodes:
            for v in sub_nodes:
                if self.graph.has_edge(u, v):
                    for k, d in self.graph.get_edge_data(u, v).items():
                        edges_list.append({"from": u, "to": v, **d})

        return {"nodes": nodes_list, "edges": edges_list}

    def clear(self):
        """Clears graph memory and storage file."""
        self.graph.clear()
        self._save()

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates topological and categorical graph statistics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Count nodes by type
        nodes_by_type: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "Unknown")
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

        # Count edges by relation type
        edges_by_type: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("type", "Unknown")
            edges_by_type[r] = edges_by_type.get(r, 0) + 1

        # Centrality
        if num_nodes > 0:
            deg_cent = nx.degree_centrality(self.graph)
            top_central = sorted(
                [{"id": nid, "name": self.graph.nodes[nid].get("name", nid), "score": round(score, 4)}
                 for nid, score in deg_cent.items()],
                key=lambda x: x["score"],
                reverse=True
            )[:10]
            density = round(nx.density(self.graph), 4)
        else:
            top_central = []
            density = 0.0

        return {
            "total_nodes": num_nodes,
            "total_edges": num_edges,
            "density": density,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "top_central_entities": top_central,
            "connected_components": nx.number_weakly_connected_components(self.graph) if num_nodes > 0 else 0
        }

