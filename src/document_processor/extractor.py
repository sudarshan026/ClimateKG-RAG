import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
import fitz  # PyMuPDF
import docx

from src.document_processor.cleaner import clean_text

@dataclass
class PageData:
    page_number: int
    text: str
    cleaned_text: str

@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    title: str
    file_type: str
    file_hash: str
    total_pages: int
    pages: List[Dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedDocument":
        return cls(**data)


def compute_file_hash(file_bytes_or_path: bytes | str | Path) -> str:
    """Computes SHA-256 hash for duplicate detection."""
    sha256 = hashlib.sha256()
    if isinstance(file_bytes_or_path, (str, Path)):
        with open(file_bytes_or_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
    else:
        sha256.update(file_bytes_or_path)
    return sha256.hexdigest()


class DocumentExtractor:
    """
    Extracts structured content and metadata from PDF, TXT, and DOCX files.
    Ensures accurate page numbering, text extraction, and duplicate detection.
    """

    @staticmethod
    def extract_from_pdf(file_path_or_bytes: str | Path | bytes, filename: str) -> ProcessedDocument:
        if isinstance(file_path_or_bytes, bytes):
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
            file_hash = compute_file_hash(file_path_or_bytes)
        else:
            file_path = Path(file_path_or_bytes)
            doc = fitz.open(str(file_path))
            file_hash = compute_file_hash(file_path)

        doc_meta = doc.metadata or {}
        extracted_title = doc_meta.get("title", "").strip()

        pages_data: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            
            # If title was not in metadata, attempt to infer from first page heading
            if not extracted_title and page_idx == 0 and cleaned:
                first_lines = [l for l in cleaned.split("\n") if len(l.strip()) > 3]
                if first_lines:
                    extracted_title = first_lines[0][:120]

            pages_data.append({
                "page_number": page_idx + 1,
                "raw_text": raw_text,
                "text": cleaned,
                "char_count": len(cleaned)
            })
            if cleaned:
                full_text_parts.append(cleaned)

        doc.close()

        final_title = extracted_title if extracted_title else Path(filename).stem.replace("_", " ").title()
        doc_id = f"doc_{file_hash[:12]}"

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            title=final_title,
            file_type="pdf",
            file_hash=file_hash,
            total_pages=len(pages_data),
            pages=pages_data,
            full_text="\n\n".join(full_text_parts),
            metadata={
                "author": doc_meta.get("author", "Unknown"),
                "subject": doc_meta.get("subject", ""),
                "keywords": doc_meta.get("keywords", ""),
                "source": filename
            }
        )

    @staticmethod
    def extract_from_txt(file_path_or_bytes: str | Path | bytes, filename: str) -> ProcessedDocument:
        if isinstance(file_path_or_bytes, bytes):
            text_content = file_path_or_bytes.decode("utf-8", errors="replace")
            file_hash = compute_file_hash(file_path_or_bytes)
        else:
            file_path = Path(file_path_or_bytes)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()
            file_hash = compute_file_hash(file_path)

        cleaned = clean_text(text_content)
        lines = [l for l in cleaned.split("\n") if len(l.strip()) > 3]
        title = lines[0][:120] if lines else Path(filename).stem.replace("_", " ").title()
        doc_id = f"doc_{file_hash[:12]}"

        pages_data = [{
            "page_number": 1,
            "raw_text": text_content,
            "text": cleaned,
            "char_count": len(cleaned)
        }]

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            file_type="txt",
            file_hash=file_hash,
            total_pages=1,
            pages=pages_data,
            full_text=cleaned,
            metadata={"source": filename}
        )

    @staticmethod
    def extract_from_docx(file_path_or_bytes: str | Path | bytes, filename: str) -> ProcessedDocument:
        import io
        if isinstance(file_path_or_bytes, bytes):
            docx_file = io.BytesIO(file_path_or_bytes)
            file_hash = compute_file_hash(file_path_or_bytes)
        else:
            docx_file = str(file_path_or_bytes)
            file_hash = compute_file_hash(Path(file_path_or_bytes))

        doc = docx.Document(docx_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        cleaned = clean_text(full_text)

        title = paragraphs[0][:120] if paragraphs else Path(filename).stem.replace("_", " ").title()
        doc_id = f"doc_{file_hash[:12]}"

        pages_data = [{
            "page_number": 1,
            "raw_text": full_text,
            "text": cleaned,
            "char_count": len(cleaned)
        }]

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            file_type="docx",
            file_hash=file_hash,
            total_pages=1,
            pages=pages_data,
            full_text=cleaned,
            metadata={"source": filename}
        )

    @classmethod
    def extract(cls, file_input: str | Path | bytes, filename: str) -> ProcessedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return cls.extract_from_pdf(file_input, filename)
        elif suffix in [".txt", ".md"]:
            return cls.extract_from_txt(file_input, filename)
        elif suffix == ".docx":
            return cls.extract_from_docx(file_input, filename)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: PDF, TXT, DOCX")

