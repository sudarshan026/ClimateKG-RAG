import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from config.settings import settings
from src.chunking.chunker import DocumentChunk
from src.retrieval.base import RetrievalResult, RetrievalResponse

logger = logging.getLogger(__name__)

class DenseRetriever:
    """
    Dense semantic retrieval using Sentence Transformers (all-MiniLM-L6-v2).
    Includes on-disk embedding cache for rapid reload without recomputing.
    """

    _model = None

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.VECTOR_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.npz"
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        if chunks:
            self.index(chunks)

    @classmethod
    def get_model(cls):
        """
        Lazy singleton loader for SentenceTransformer.
        Supports initial download when online, followed by seamless offline execution.
        """
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = settings.EMBEDDING_MODEL_NAME
            try:
                # 1. Attempt offline load from local cache first
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                cls._model = SentenceTransformer(model_name)
                logger.info(f"Loaded SentenceTransformer '{model_name}' from local cache.")
            except Exception as e_offline:
                logger.info(f"Local cache miss or offline check failed ({e_offline}). Attempting online fetch for '{model_name}'...")
                try:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    cls._model = SentenceTransformer(model_name)
                    # Once downloaded, enable offline protection for future calls
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    logger.info(f"Successfully downloaded and cached '{model_name}'.")
                except Exception as e_online:
                    logger.error(f"Failed to load or download SentenceTransformer '{model_name}': {e_online}")
                    raise RuntimeError(
                        f"SentenceTransformer '{model_name}' could not be loaded from cache or downloaded. "
                        f"Ensure model is cached or network access is available for initial download. Error: {e_online}"
                    )
        return cls._model

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Loads cached embeddings dictionary {chunk_id: embedding_vector}."""
        if not self.cache_file.exists():
            return {}
        try:
            data = np.load(self.cache_file)
            cached = {}
            for cid in data.files:
                cached[cid] = data[cid]
            return cached
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
            return {}

    def _save_cache(self, cache_dict: Dict[str, np.ndarray]):
        """Saves embedding dictionary to npz archive."""
        try:
            np.savez_compressed(self.cache_file, **cache_dict)
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not self.chunks:
            self.embeddings = None
            self.chunk_ids = []
            return

        self.chunk_ids = [c.chunk_id for c in self.chunks]
        cached_dict = self._load_cache()
        missing_indices = []
        missing_texts = []

        for idx, chunk in enumerate(self.chunks):
            if chunk.chunk_id not in cached_dict:
                missing_indices.append(idx)
                missing_texts.append(chunk.text)

        if missing_texts:
            model = self.get_model()
            logger.info(f"Encoding {len(missing_texts)} new chunks...")
            new_embeddings = model.encode(missing_texts, show_progress_bar=False, normalize_embeddings=True)
            for idx, emb in zip(missing_indices, new_embeddings):
                cached_dict[self.chunks[idx].chunk_id] = emb.astype(np.float32)
            self._save_cache(cached_dict)

        # Assemble unified embedding matrix matching self.chunks
        emb_list = [cached_dict[cid] for cid in self.chunk_ids]
        self.embeddings = np.vstack(emb_list).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        start_time = time.perf_counter()

        if not self.chunks or self.embeddings is None or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return RetrievalResponse(
                query=query,
                method="Dense Semantic",
                results=[],
                latency_ms=round(latency, 2),
                total_chunks_searched=len(self.chunks)
            )

        model = self.get_model()
        query_embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]

        # Cosine similarity for normalized vectors is simply the dot product
        scores = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[RetrievalResult] = []

        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
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
                method="Dense Semantic",
                metadata=chunk.metadata
            ))

        latency = (time.perf_counter() - start_time) * 1000
        return RetrievalResponse(
            query=query,
            method="Dense Semantic",
            results=results,
            latency_ms=round(latency, 2),
            total_chunks_searched=len(self.chunks)
        )
