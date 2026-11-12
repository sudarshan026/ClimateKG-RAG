from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    page_number: int
    text: str
    score: float
    rank: int = 1
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RetrievalResponse:
    query: str
    method: str
    results: List[RetrievalResult]
    latency_ms: float
    total_chunks_searched: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "latency_ms": self.latency_ms,
            "total_chunks_searched": self.total_chunks_searched,
            "results": [r.to_dict() for r in self.results]
        }

