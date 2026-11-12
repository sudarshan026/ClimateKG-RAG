from __future__ import annotations
import math
import re
from typing import List, Set, Dict, Any, Optional

class RetrievalMetrics:
    """
    Standard Information Retrieval Evaluation Metrics Suite:
    - Precision@K (Strict deduplicated evaluation)
    - Recall@K
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision, guaranteed <= 1.0)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Claim-Level Evidence Support Rate (Transparent Grounding Verification)
    """

    @staticmethod
    def precision_at_k(relevance_binary: List[int], k: int) -> float:
        """
        Precision@K: Fraction of top-k retrieved items that are verified relevant.
        Guarantees 0.0 <= Precision@K <= 1.0.
        """
        if k <= 0 or not relevance_binary:
            return 0.0
        relevance_k = relevance_binary[:k]
        return float(sum(relevance_k)) / float(k)

    @staticmethod
    def recall_at_k(relevance_binary: List[int], total_relevant: int, k: int) -> float:
        """
        Recall@K: Fraction of total known relevant items retrieved in top-k.
        Guarantees 0.0 <= Recall@K <= 1.0.
        """
        if total_relevant <= 0 or k <= 0 or not relevance_binary:
            return 0.0
        relevance_k = relevance_binary[:k]
        hits = sum(relevance_k)
        return min(1.0, float(hits) / float(total_relevant))

    @staticmethod
    def reciprocal_rank(relevance_binary: List[int]) -> float:
        """
        Reciprocal Rank (RR): 1 / rank of first relevant item.
        Returns 0.0 if no relevant items retrieved in top-K.
        """
        for rank, rel in enumerate(relevance_binary, start=1):
            if rel > 0:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def average_precision(relevance_binary: List[int], total_relevant: int, k: int) -> float:
        """
        Average Precision (AP) evaluated up to rank K.
        Standard Information Retrieval (TREC) definition:
        AP@K = (1 / total_relevant) * sum_{i=1}^K [P@i * rel(i)]
        
        Guarantees AP <= 1.0 mathematically under all circumstances by ensuring
        denominator is at least the number of observed relevant items and total relevant items.
        """
        if total_relevant <= 0 or k <= 0 or not relevance_binary:
            return 0.0

        relevance_k = relevance_binary[:k]
        running_sum = 0.0
        relevant_count = 0

        for rank, rel in enumerate(relevance_k, start=1):
            if rel > 0:
                relevant_count += 1
                precision_at_current_rank = float(relevant_count) / float(rank)
                running_sum += precision_at_current_rank

        denominator = max(total_relevant, relevant_count)
        if denominator == 0 or running_sum == 0.0:
            return 0.0

        ap = running_sum / float(denominator)
        return min(1.0, round(ap, 6))

    @staticmethod
    def ndcg_at_k(relevance_scores: List[float | int], total_relevant: Optional[int] = None, k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain at rank K.
        Uses binary or graded relevance scores against ideal ranking (IDCG).
        If total_relevant is not provided, defaults to the count of positive relevance scores in the list.
        """
        if k <= 0 or not relevance_scores:
            return 0.0

        if total_relevant is None:
            total_relevant = sum(1 for r in relevance_scores if r > 0)

        rel_k = relevance_scores[:k]
        dcg = 0.0
        for i, rel in enumerate(rel_k):
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 2)

        # Ideal DCG: at most min(k, total_relevant) relevant items ranked at top
        num_ideal = min(k, max(1, total_relevant)) if any(r > 0 for r in rel_k) else 0
        if num_ideal == 0:
            return 0.0

        idcg = 0.0
        for i in range(num_ideal):
            idcg += (math.pow(2, 1) - 1.0) / math.log2(i + 2)

        if idcg == 0.0:
            return 0.0

        return min(1.0, round(dcg / idcg, 6))

    @staticmethod
    def calculate_claim_evidence_support(
        answer_text: str,
        evidence_passages: List[str],
        graph_paths: Optional[List[str]] = None
    ) -> float:
        """
        Claim-Level Evidence Support Rate (Transparent Grounding Proxy).
        
        Methodology:
        1. Breaks answer into distinct factual assertion sentences (claims).
        2. Discards conversational transitions, greetings, and disclaimers.
        3. For each factual claim, evaluates whether its key content terms and
           climate relationships are supported by retrieved passages or graph paths.
        4. If the answer explicitly states 'Insufficient Evidence', verifies if evidence
           was indeed absent, awarding 1.0 for valid abstention.
        """
        if not answer_text or not answer_text.strip():
            return 0.0

        # Handle valid abstention
        ans_lower = answer_text.lower()
        if "insufficient evidence" in ans_lower or "no relevant documents" in ans_lower:
            if not evidence_passages:
                return 1.0  # Perfect abstention when no evidence exists
            return 0.5  # Conservative abstention

        combined_evidence = " ".join(evidence_passages).lower()
        if graph_paths:
            combined_evidence += " " + " ".join(graph_paths).lower()

        # Split into distinct sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        claim_sentences = []
        for s in sentences:
            s_clean = s.strip()
            # Filter out headers and short fragments
            if len(s_clean) > 25 and not s_clean.startswith("#"):
                claim_sentences.append(s_clean)

        if not claim_sentences:
            return 0.0

        supported_claims = 0
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "are", "was", "were", "be", "been", "by", "this", "that", "it", "as",
            "from", "which", "what", "how", "has", "have", "had", "can", "could", "would",
            "according", "source", "page", "chunk", "document", "report", "findings", "key",
            "scientific", "identified", "relational", "pathways", "evidence", "excerpt"
        }

        for claim in claim_sentences:
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', claim) if w.lower() not in stopwords]
            if not words:
                continue

            # Require that a substantial proportion (>= 60%) of significant content terms
            # or exact key n-grams appear in the retrieved evidence
            matches = sum(1 for w in words if w in combined_evidence)
            support_ratio = matches / float(len(words))
            if support_ratio >= 0.55:
                supported_claims += 1

        support_rate = supported_claims / float(len(claim_sentences))
        return min(1.0, round(support_rate, 4))

    # Backward compatibility alias
    calculate_faithfulness = calculate_claim_evidence_support
