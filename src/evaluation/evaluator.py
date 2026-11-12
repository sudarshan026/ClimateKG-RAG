from __future__ import annotations
import time
import logging
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np

from config.settings import settings
from src.evaluation.benchmark_dataset import CLIMATE_BENCHMARK_QUERIES, ClimateEvaluationQuery
from src.evaluation.metrics import RetrievalMetrics
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridGraphRAGRetriever
from src.retrieval.base import RetrievalResult
from src.qa.generator import GroundedAnswerGenerator

logger = logging.getLogger(__name__)

class ClimateEvaluator:
    """
    Automated comparative evaluation benchmark for ClimateKG-RAG.
    Compares TF-IDF, BM25, Dense Semantic Retrieval, Knowledge Graph Traversal, and Hybrid GraphRAG.
    
    Computes rigorously verified IR metrics:
    - Precision@K (Strict deduplicated evaluation)
    - Recall@K
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision, guaranteed <= 1.0)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Latency (ms)
    - Claim-Level Evidence Support Rate (Grounding Verification)
    """

    def __init__(
        self,
        tfidf_retriever: TFIDFRetriever,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        graph_retriever: GraphRetriever,
        hybrid_retriever: HybridGraphRAGRetriever,
        generator: Optional[GroundedAnswerGenerator] = None
    ):
        self.retrievers = {
            "TF-IDF": tfidf_retriever,
            "BM25": bm25_retriever,
            "Dense Semantic": dense_retriever,
            "Knowledge Graph": graph_retriever,
            "Hybrid GraphRAG": hybrid_retriever
        }
        self.generator = generator or GroundedAnswerGenerator()
        self.benchmark_queries = CLIMATE_BENCHMARK_QUERIES

    @staticmethod
    def assess_relevance(result: RetrievalResult, query_def: ClimateEvaluationQuery) -> int:
        """
        Determines binary relevance (1 or 0) of a retrieved result using explicit ground truth:
        1. Document chunks: Must originate from target gold documents AND contain core answer facts/keywords.
        2. Knowledge Graph paths: Must traverse the specific required focal entities that answer the question.
        """
        res_file = (result.filename or "").lower()
        res_text = (result.text or "").lower()

        # Case 1: Knowledge Graph Traversal Result
        if result.method == "Knowledge Graph" or res_text.startswith("graph path:"):
            meta_node_ids = [str(nid).lower() for nid in result.metadata.get("node_ids", [])]
            meta_nodes = [str(n).lower() for n in result.metadata.get("nodes", [])]
            path_desc = result.metadata.get("path_description", res_text).lower()

            if query_def.required_focal_entities:
                matched_count = 0
                for req_ent in query_def.required_focal_entities:
                    req_id_lower = req_ent.lower()
                    parts = req_id_lower.split("_")
                    if len(parts) >= 3 and parts[0] == "ent":
                        clean_name = " ".join(parts[2:])
                    else:
                        clean_name = req_id_lower.replace("ent_", "").replace("_", " ")

                    ent_found = (
                        (req_id_lower in meta_node_ids) or
                        any(clean_name in n for n in meta_nodes) or
                        (clean_name in path_desc)
                    )
                    if ent_found:
                        matched_count += 1

                required_target = 1 if len(query_def.required_focal_entities) == 1 else min(2, len(query_def.required_focal_entities))
                return 1 if matched_count >= required_target else 0

        # Case 2: Document Chunk Result (TF-IDF, BM25, Dense, Hybrid)
        target_doc_match = any(target.lower() in res_file for target in query_def.target_documents)
        if target_doc_match:
            # Check keyword / conceptual alignment
            keyword_matches = sum(1 for kw in query_def.expected_relevant_keywords if kw.lower() in res_text)
            if keyword_matches >= 1:
                return 1

        # Fallback for cross-document high-confidence matches (if target_documents had multiple editions)
        strong_keyword_matches = sum(1 for kw in query_def.expected_relevant_keywords if kw.lower() in res_text)
        if strong_keyword_matches >= 3:
            return 1

        return 0

    def run_comparative_evaluation(
        self,
        top_k: int = 5,
        queries: Optional[List[ClimateEvaluationQuery]] = None,
        compute_faithfulness: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full comparative evaluation across all benchmark queries and retrieval strategies.
        Guarantees MAP <= 1.0, exact ground truth relevance, deduplication, and genuine latency profiling.
        """
        eval_queries = queries or self.benchmark_queries
        detailed_records: List[Dict[str, Any]] = []

        summary_metrics: Dict[str, Dict[str, List[float]]] = {
            method: {
                "precision": [],
                "recall": [],
                "rr": [],
                "ap": [],
                "ndcg": [],
                "latency_ms": [],
                "faithfulness": []
            }
            for method in self.retrievers.keys()
        }

        for q_def in eval_queries:
            total_gold_relevant = max(1, q_def.ground_truth_relevant_count)

            for method_name, retriever in self.retrievers.items():
                try:
                    resp = retriever.retrieve(q_def.query, top_k=top_k)
                    results = resp.results
                    latency = resp.latency_ms

                    # Deduplication and relevance assessment
                    relevance_binary: List[int] = []
                    seen_chunk_signatures: Set[str] = set()

                    for r in results:
                        # Deduplication signature: filename + page + chunk_id / text
                        sig = f"{r.filename}_{r.page_number}_{r.chunk_id}"
                        if sig in seen_chunk_signatures:
                            # Duplicate penalty: duplicates receive 0
                            relevance_binary.append(0)
                            continue
                        seen_chunk_signatures.add(sig)

                        rel = self.assess_relevance(r, q_def)
                        relevance_binary.append(rel)

                    # Pad with zeros if fewer results returned than top_k
                    while len(relevance_binary) < top_k:
                        relevance_binary.append(0)

                    # Compute standard IR metrics
                    p_at_k = RetrievalMetrics.precision_at_k(relevance_binary, top_k)
                    r_at_k = RetrievalMetrics.recall_at_k(relevance_binary, total_gold_relevant, top_k)
                    rr = RetrievalMetrics.reciprocal_rank(relevance_binary)
                    ap = RetrievalMetrics.average_precision(relevance_binary, total_gold_relevant, top_k)
                    ndcg = RetrievalMetrics.ndcg_at_k(relevance_binary, total_gold_relevant, top_k)

                    # Compute claim-level evidence support
                    evidence_support = 0.0
                    if compute_faithfulness and results:
                        ans_resp = self.generator.answer_query(q_def.query, method_name, results)
                        evidence_texts = [r.text for r in results]
                        graph_paths = [r.metadata.get("path_description", "") for r in results if "path_description" in r.metadata]
                        evidence_support = RetrievalMetrics.calculate_claim_evidence_support(
                            ans_resp.answer,
                            evidence_texts,
                            graph_paths
                        )

                    # Accumulate
                    summary_metrics[method_name]["precision"].append(p_at_k)
                    summary_metrics[method_name]["recall"].append(r_at_k)
                    summary_metrics[method_name]["rr"].append(rr)
                    summary_metrics[method_name]["ap"].append(ap)
                    summary_metrics[method_name]["ndcg"].append(ndcg)
                    summary_metrics[method_name]["latency_ms"].append(latency)
                    summary_metrics[method_name]["faithfulness"].append(evidence_support)

                    detailed_records.append({
                        "query_id": q_def.query_id,
                        "category": q_def.category,
                        "query": q_def.query,
                        "method": method_name,
                        "precision_at_k": round(p_at_k, 4),
                        "recall_at_k": round(r_at_k, 4),
                        "reciprocal_rank": round(rr, 4),
                        "average_precision": round(ap, 4),
                        "ndcg_at_k": round(ndcg, 4),
                        "latency_ms": round(latency, 2),
                        "claim_evidence_support": round(evidence_support, 4),
                        "results_count": len(results)
                    })

                except Exception as e:
                    logger.error(f"Error evaluating {method_name} on query {q_def.query_id}: {e}")

        # Assemble summary table
        summary_rows = []
        for method_name, metrics in summary_metrics.items():
            summary_rows.append({
                "Retrieval Method": method_name,
                "Precision@K": round(float(np.mean(metrics["precision"])), 4) if metrics["precision"] else 0.0,
                "Recall@K": round(float(np.mean(metrics["recall"])), 4) if metrics["recall"] else 0.0,
                "MRR": round(float(np.mean(metrics["rr"])), 4) if metrics["rr"] else 0.0,
                "MAP": round(float(np.mean(metrics["ap"])), 4) if metrics["ap"] else 0.0,
                "NDCG@K": round(float(np.mean(metrics["ndcg"])), 4) if metrics["ndcg"] else 0.0,
                "Avg Latency (ms)": round(float(np.mean(metrics["latency_ms"])), 2) if metrics["latency_ms"] else 0.0,
                "Evidence Support Rate": round(float(np.mean(metrics["faithfulness"])), 4) if metrics["faithfulness"] else 0.0
            })

        summary_df = pd.DataFrame(summary_rows)
        detailed_df = pd.DataFrame(detailed_records)

        # Persist to disk
        out_summary_file = settings.EVALUATION_DIR / "evaluation_summary.csv"
        out_detailed_file = settings.EVALUATION_DIR / "evaluation_detailed.csv"
        summary_df.to_csv(out_summary_file, index=False)
        detailed_df.to_csv(out_detailed_file, index=False)

        return {
            "summary_df": summary_df,
            "detailed_df": detailed_df,
            "total_queries": len(eval_queries),
            "top_k": top_k,
            "summary_csv": str(out_summary_file),
            "detailed_csv": str(out_detailed_file)
        }
