from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from src.retrieval.base import RetrievalResult

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining ranked lists from
    multiple heterogeneous retrieval strategies (BM25, Dense, Knowledge Graph).
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[RetrievalResult]],
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 5
    ) -> List[RetrievalResult]:
        """
        Fuses multiple ranked lists using RRF.
        
        Args:
            ranked_lists: Dict mapping method_name -> List[RetrievalResult]
            weights: Optional Dict mapping method_name -> weight multiplier (default 1.0)
            top_n: Number of top fused results to return
            
        Returns:
            Ranked list of unified RetrievalResult objects with fused score and provenance
        """
        if not ranked_lists:
            return []

        weights = weights or {}
        fused_scores: Dict[str, float] = defaultdict(float)
        item_registry: Dict[str, RetrievalResult] = {}
        contributing_methods: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

        for method_name, results in ranked_lists.items():
            weight = weights.get(method_name, 1.0)
            seen_in_method: Set[str] = set()
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk_id
                # Avoid duplicate chunk scoring within the same retriever list
                if chunk_id in seen_in_method:
                    continue
                seen_in_method.add(chunk_id)

                # RRF scoring formula: weight / (k + rank)
                rrf_increment = weight / (self.k + rank)
                fused_scores[chunk_id] += rrf_increment
                contributing_methods[chunk_id].append((method_name, rank, item.score))

                if chunk_id not in item_registry:
                    item_registry[chunk_id] = item
                else:
                    # If this item contains graph path information, preserve it in the registry metadata
                    if "path_description" in item.metadata:
                        item_registry[chunk_id].metadata["path_description"] = item.metadata["path_description"]

        # Sort items by fused score descending
        sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        fused_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_n], start=1):
            base_item = item_registry[cid]
            fused_score = fused_scores[cid]
            meta = dict(base_item.metadata)
            meta["contributing_methods"] = contributing_methods[cid]
            meta["fused_rrf_score"] = round(fused_score, 6)

            fused_results.append(RetrievalResult(
                chunk_id=base_item.chunk_id,
                doc_id=base_item.doc_id,
                doc_title=base_item.doc_title,
                filename=base_item.filename,
                page_number=base_item.page_number,
                text=base_item.text,
                score=round(fused_score, 6),
                rank=rank,
                method="Hybrid-RRF",
                metadata=meta
            ))

        return fused_results

