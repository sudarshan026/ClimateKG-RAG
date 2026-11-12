from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from src.document_processor.extractor import ProcessedDocument

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    page_number: int
    text: str
    word_count: int
    char_start: int
    char_end: int
    chunk_index: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class TextChunker:
    """
    Sentence-boundary-aware document chunker with configurable word size and overlap.
    Preserves document and page level provenance for rigorous evidence grounding.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Splits text into sentences using punctuation boundaries."""
        # Handles periods, question marks, exclamation points, followed by whitespace/capital letter
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_page(
        self,
        page_text: str,
        page_number: int,
        doc_id: str,
        doc_title: str,
        filename: str,
        base_chunk_idx: int = 0
    ) -> List[DocumentChunk]:
        """Chunks a single page's text into overlapping sentence-aware segments."""
        if not page_text or not page_text.strip():
            return []

        sentences = self.split_into_sentences(page_text)
        if not sentences:
            sentences = [page_text]

        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_word_count = 0
        char_offset = 0
        chunk_idx = base_chunk_idx

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences)
                c_start = page_text.find(chunk_str, char_offset)
                if c_start == -1:
                    c_start = char_offset
                c_end = c_start + len(chunk_str)
                char_offset = c_start

                chunk_id = f"{doc_id}_p{page_number}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    filename=filename,
                    page_number=page_number,
                    text=chunk_str,
                    word_count=len(chunk_str.split()),
                    char_start=c_start,
                    char_end=c_end,
                    chunk_index=chunk_idx,
                    metadata={
                        "source": filename,
                        "doc_title": doc_title,
                        "page": page_number
                    }
                ))
                chunk_idx += 1

                # Retain overlap sentences
                overlap_sentences: List[str] = []
                overlap_words = 0
                for s in reversed(current_sentences):
                    s_words = len(s.split())
                    if overlap_words + s_words <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                current_sentences = overlap_sentences
                current_word_count = overlap_words

            current_sentences.append(sentence)
            current_word_count += sentence_words

        # Append trailing sentences
        if current_sentences:
            chunk_str = " ".join(current_sentences)
            c_start = page_text.find(chunk_str, char_offset)
            if c_start == -1:
                c_start = char_offset
            c_end = c_start + len(chunk_str)

            chunk_id = f"{doc_id}_p{page_number}_c{chunk_idx}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                filename=filename,
                page_number=page_number,
                text=chunk_str,
                word_count=len(chunk_str.split()),
                char_start=c_start,
                char_end=c_end,
                chunk_index=chunk_idx,
                metadata={
                    "source": filename,
                    "doc_title": doc_title,
                    "page": page_number
                }
            ))

        return chunks

    def chunk_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """Processes all pages of a document into a sequential list of DocumentChunk objects."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in doc.pages:
            p_num = page["page_number"]
            p_text = page["text"]
            page_chunks = self.chunk_page(
                page_text=p_text,
                page_number=p_num,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                filename=doc.filename,
                base_chunk_idx=global_chunk_idx
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks
