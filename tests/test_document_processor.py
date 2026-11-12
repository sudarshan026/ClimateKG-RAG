import pytest
import tempfile
from pathlib import Path
from src.document_processor.cleaner import clean_text
from src.document_processor.extractor import DocumentExtractor, compute_file_hash
from src.document_processor.store import DocumentStore
from src.chunking.chunker import TextChunker, DocumentChunk

def test_clean_text_basic():
    dirty = "Climate  change   is  real.\r\nIt affects    bio-\ndiversity.\n\n\nNew paragraph."
    cleaned = clean_text(dirty)
    assert "biodiversity" in cleaned
    assert "  " not in cleaned
    assert "New paragraph." in cleaned

def test_txt_extraction_and_hash():
    sample_text = "Global warming causes extreme weather events across South Asia."
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(sample_text)
        temp_path = f.name

    try:
        doc = DocumentExtractor.extract(temp_path, "sample.txt")
        assert doc.filename == "sample.txt"
        assert doc.total_pages == 1
        assert "Global warming" in doc.full_text
        assert doc.file_hash == compute_file_hash(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

def test_document_store_and_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(storage_dir=Path(tmpdir))
        
        sample_text = "Renewable energy mitigation strategies."
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(sample_text)
            temp_path = f.name
            
        try:
            doc1 = DocumentExtractor.extract(temp_path, "report1.txt")
            chunker = TextChunker(chunk_size=100, chunk_overlap=20)
            chunks1 = chunker.chunk_document(doc1)
            
            added1 = store.add_document(doc1, chunks1)
            assert added1 is True
            assert store.get_statistics()["total_documents"] == 1
            assert store.get_statistics()["total_chunks"] == len(chunks1)
            
            # Attempt duplicate insertion
            doc2 = DocumentExtractor.extract(temp_path, "report2_duplicate.txt")
            chunks2 = chunker.chunk_document(doc2)
            added2 = store.add_document(doc2, chunks2)
            assert added2 is False  # Duplicate detected!
            assert store.get_statistics()["total_documents"] == 1
            
            # Deletion
            deleted = store.delete_document(doc1.doc_id)
            assert deleted is True
            assert store.get_statistics()["total_documents"] == 0
            assert store.get_statistics()["total_chunks"] == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

