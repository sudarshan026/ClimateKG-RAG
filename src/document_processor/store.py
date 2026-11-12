from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from config.settings import settings
from src.document_processor.extractor import ProcessedDocument
from src.chunking.chunker import DocumentChunk

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Persistent document and chunk storage manager.
    Handles indexing, duplicate detection, metadata retrieval, and deletion.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.PROCESSED_DATA_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.docs_file = self.storage_dir / "documents.json"
        self.chunks_file = self.storage_dir / "chunks.json"
        self.documents: Dict[str, ProcessedDocument] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self._load()

    def _load(self):
        """Loads indexed documents and chunks from disk if they exist."""
        if self.docs_file.exists():
            try:
                with open(self.docs_file, "r", encoding="utf-8") as f:
                    docs_raw = json.load(f)
                    self.documents = {k: ProcessedDocument.from_dict(v) for k, v in docs_raw.items()}
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
                self.documents = {}

        if self.chunks_file.exists():
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    chunks_raw = json.load(f)
                    self.chunks = {k: DocumentChunk.from_dict(v) for k, v in chunks_raw.items()}
            except Exception as e:
                logger.error(f"Error loading chunks: {e}")
                self.chunks = {}

    def _save(self):
        """Persists indexed documents and chunks to disk."""
        with open(self.docs_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self.documents.items()}, f, indent=2, ensure_ascii=False)

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self.chunks.items()}, f, indent=2, ensure_ascii=False)

    def is_duplicate(self, file_hash: str) -> Optional[str]:
        """Checks if a document with the given content hash has already been indexed."""
        for doc_id, doc in self.documents.items():
            if doc.file_hash == file_hash:
                return doc_id
        return None

    def add_document(self, doc: ProcessedDocument, chunks: List[DocumentChunk]) -> bool:
        """
        Adds a document and its chunks to the store.
        If file hash already exists, returns False (duplicate).
        """
        existing_id = self.is_duplicate(doc.file_hash)
        if existing_id:
            logger.info(f"Duplicate document detected: {doc.filename} matches {existing_id}")
            return False

        self.documents[doc.doc_id] = doc
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

        self._save()
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document and its associated chunks from the store."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        # Remove all chunks associated with this doc_id
        chunk_keys_to_remove = [cid for cid, chk in self.chunks.items() if chk.doc_id == doc_id]
        for cid in chunk_keys_to_remove:
            del self.chunks[cid]

        self._save()
        return True

    def get_document(self, doc_id: str) -> Optional[ProcessedDocument]:
        return self.documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def get_all_documents(self) -> List[ProcessedDocument]:
        return list(self.documents.values())

    def get_all_chunks(self) -> List[DocumentChunk]:
        return list(self.chunks.values())

    def get_chunks_for_document(self, doc_id: str) -> List[DocumentChunk]:
        return [chk for chk in self.chunks.values() if chk.doc_id == doc_id]

    def clear(self):
        """Wipes all documents and chunks from the store."""
        self.documents.clear()
        self.chunks.clear()
        self._save()

    def get_statistics(self) -> Dict[str, Any]:
        """Computes live storage statistics."""
        total_docs = len(self.documents)
        total_chunks = len(self.chunks)
        total_pages = sum(d.total_pages for d in self.documents.values())
        total_words = sum(c.word_count for c in self.chunks.values())
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "total_words": total_words,
            "avg_chunk_size": round(total_words / total_chunks, 1) if total_chunks > 0 else 0
        }
