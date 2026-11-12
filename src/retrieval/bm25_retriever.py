import time
import re
from typing import List, Optional
from rank_bm25 import BM25Okapi

from src.chunking.chunker import DocumentChunk
from src.retrieval.base import RetrievalResult, RetrievalResponse

class BM25Retriever:
    """
    Lexical retrieval using the industry-standard BM25Okapi algorithm.
    """

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.index(chunks)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, fast lowercase alphanumeric tokenization."""
        return re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not self.chunks:
            self.bm25 = None
            return

        tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        start_time = time.perf_counter()

        if not self.chunks or self.bm25 is None or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return RetrievalResponse(
                query=query,
                method="BM25",
                results=[],
                latency_ms=round(latency, 2),
                total_chunks_searched=len(self.chunks)
            )

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Standard BM25 scores can be <= 0 in tiny corpora (N <= 2) due to Robertson IDF = ln(1) = 0.
        # Fall back to term frequency overlap if max score <= 0.
        max_score = float(max(scores)) if len(scores) > 0 else 0.0
        indexed_scores = []

        if max_score > 0.0:
            indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0]
        else:
            # Fallback for small corpus: count term overlap
            q_set = set(tokenized_query)
            for idx, c in enumerate(self.chunks):
                c_toks = set(self.tokenize(c.text))
                overlap = len(q_set.intersection(c_toks))
                if overlap > 0:
                    indexed_scores.append((idx, float(overlap)))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=round(float(score), 4),
                rank=rank,
                method="BM25",
                metadata=chunk.metadata
            ))

        latency = (time.perf_counter() - start_time) * 1000
        return RetrievalResponse(
            query=query,
            method="BM25",
            results=results,
            latency_ms=round(latency, 2),
            total_chunks_searched=len(self.chunks)
        )
