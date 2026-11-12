import pytest
from src.chunking.chunker import TextChunker
from src.document_processor.extractor import ProcessedDocument

def test_chunker_sentence_boundary():
    chunker = TextChunker(chunk_size=15, chunk_overlap=5)
    sentences = [
        "First sentence discussing carbon dioxide emissions.",
        "Second sentence discussing methane leaks in agriculture.",
        "Third sentence reviewing solar energy policies in India.",
        "Fourth sentence concluding the mitigation pathways."
    ]
    full_text = " ".join(sentences)
    doc = ProcessedDocument(
        doc_id="doc_123",
        filename="test.txt",
        title="Test Document",
        file_type="txt",
        file_hash="hash123",
        total_pages=1,
        pages=[{"page_number": 1, "text": full_text, "raw_text": full_text, "char_count": len(full_text)}],
        full_text=full_text
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.doc_id == "doc_123"
        assert chunk.page_number == 1
        assert chunk.chunk_id.startswith("doc_123_p1_c")
        assert len(chunk.text) > 0
        assert chunk.metadata["source"] == "test.txt"

def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)

