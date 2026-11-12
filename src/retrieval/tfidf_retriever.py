import time
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.chunker import DocumentChunk
from src.retrieval.base import RetrievalResult, RetrievalResponse

class TFIDFRetriever:
    """
    Lexical retrieval using Scikit-Learn's TF-IDF Vectorizer and Cosine Similarity.
    """

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not self.chunks:
            self.tfidf_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        start_time = time.perf_counter()

        if not self.chunks or self.tfidf_matrix is None or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return RetrievalResponse(
                query=query,
                method="TF-IDF",
                results=[],
                latency_ms=round(latency, 2),
                total_chunks_searched=len(self.chunks)
            )

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[RetrievalResult] = []

        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=round(score, 4),
                rank=rank,
                method="TF-IDF",
                metadata=chunk.metadata
            ))

        latency = (time.perf_counter() - start_time) * 1000
        return RetrievalResponse(
            query=query,
            method="TF-IDF",
            results=results,
            latency_ms=round(latency, 2),
            total_chunks_searched=len(self.chunks)
        )

