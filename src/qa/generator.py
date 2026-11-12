import re
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import requests

from config.settings import settings
from src.retrieval.base import RetrievalResult
from src.qa.prompt_templates import build_grounded_qa_prompt

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    source_index: int
    filename: str
    doc_title: str
    page_number: int
    chunk_id: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnswerResponse:
    query: str
    answer: str
    retrieval_method: str
    citations: List[Citation]
    graph_paths: List[str]
    retrieved_results: List[RetrievalResult]
    confidence: str  # "High", "Medium", "Low", "Insufficient Evidence"
    confidence_score: float
    generator_type: str  # "Ollama LLM" or "Deterministic Synthesis"
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieval_method": self.retrieval_method,
            "citations": [c.to_dict() for c in self.citations],
            "graph_paths": self.graph_paths,
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "generator_type": self.generator_type,
            "latency_ms": self.latency_ms
        }


class GroundedAnswerGenerator:
    """
    Evidence-grounded question answering engine.
    Uses local Ollama LLM when available; otherwise provides a rigorous,
    deterministic extractive synthesis engine with exact provenance tracking.
    """

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        self.ollama_url = ollama_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self._ollama_checked = False
        self._ollama_available = False

    def is_ollama_available(self) -> bool:
        if not settings.USE_OLLAMA_IF_AVAILABLE:
            return False
        if self._ollama_checked:
            return self._ollama_available
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1)
            self._ollama_available = (resp.status_code == 200)
        except Exception:
            self._ollama_available = False
        self._ollama_checked = True
        return self._ollama_available

    def generate_with_ollama(
        self,
        query: str,
        results: List[RetrievalResult],
        graph_paths: List[str]
    ) -> Optional[str]:
        """Calls local Ollama instance with grounded prompt."""
        passages = [r.to_dict() for r in results]
        prompt = build_grounded_qa_prompt(query, passages, graph_paths)

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9}
                },
                timeout=settings.OLLAMA_TIMEOUT
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama generation request failed: {e}")
        return None

    def generate_deterministic_synthesis(
        self,
        query: str,
        results: List[RetrievalResult],
        graph_paths: List[str]
    ) -> str:
        """
        Extractive synthesis engine that summarizes verified document passages
        and multi-hop graph paths without fabricating claims.
        """
        if not results:
            return (
                "**Insufficient Evidence:** No relevant documents or Knowledge Graph records "
                "were found matching this query in the ingested corpus."
            )

        q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', query)]

        # Extract most informative sentences across top chunks
        extracted_facts: List[Dict[str, Any]] = []
        for r in results[:4]:
            # Clean passage text
            clean_p = r.text
            if clean_p.startswith("GRAPH PATH:"):
                # Handle graph path passage
                continue

            sentences = re.split(r'(?<=[.!?])\s+', clean_p)
            for s in sentences:
                s_strip = s.strip()
                if len(s_strip) < 25:
                    continue
                s_lower = s_strip.lower()
                matches = sum(1 for w in q_words if w in s_lower)
                if matches > 0:
                    extracted_facts.append({
                        "sentence": s_strip,
                        "matches": matches,
                        "filename": r.filename,
                        "page": r.page_number,
                        "chunk_id": r.chunk_id
                    })

        # Sort facts by term overlap
        extracted_facts.sort(key=lambda x: x["matches"], reverse=True)

        # Build synthesis text
        response_sections = []

        # 1. Summary of direct findings
        if extracted_facts:
            response_sections.append("### Key Scientific Findings\n")
            seen_sentences = set()
            count = 0
            for fact in extracted_facts:
                if fact["sentence"] in seen_sentences:
                    continue
                seen_sentences.add(fact["sentence"])
                response_sections.append(
                    f"- {fact['sentence']} `[Source: {fact['filename']}, Page {fact['page']}]`"
                )
                count += 1
                if count >= 4:
                    break
        else:
            # Fall back to using top chunk snippet directly
            top_chunk = results[0]
            snippet = " ".join(top_chunk.text.split()[:80]) + "..."
            response_sections.append(
                f"### Evidence Excerpt\n{snippet} `[Source: {top_chunk.filename}, Page {top_chunk.page_number}]`"
            )

        # 2. Relational Knowledge Graph Findings (Multi-hop)
        if graph_paths:
            response_sections.append("\n### Knowledge Graph Relational Pathways\n")
            response_sections.append("Multi-hop relational traversal identified the following climate mechanisms:")
            for p in graph_paths[:4]:
                response_sections.append(f"- **Pathway:** {p}")

        return "\n".join(response_sections)

    def answer_query(
        self,
        query: str,
        retrieval_method: str,
        results: List[RetrievalResult]
    ) -> AnswerResponse:
        """
        Generates a verified, citation-backed answer for the given query and retrieval results.
        """
        start_time = time.perf_counter()

        # Extract graph paths from results
        graph_paths = []
        for r in results:
            if "path_description" in r.metadata:
                graph_paths.append(r.metadata["path_description"])
            elif r.text.startswith("GRAPH PATH:"):
                first_line = r.text.split("\n")[0].replace("GRAPH PATH: ", "")
                graph_paths.append(first_line)

        # Extract citations
        citations: List[Citation] = []
        for idx, r in enumerate(results, start=1):
            snippet = r.text[:180] + "..." if len(r.text) > 180 else r.text
            citations.append(Citation(
                source_index=idx,
                filename=r.filename,
                doc_title=r.doc_title,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
                snippet=snippet
            ))

        # Determine confidence
        if not results:
            confidence = "Insufficient Evidence"
            confidence_score = 0.0
        else:
            top_score = results[0].score
            if top_score > 0.4 or (retrieval_method == "Hybrid GraphRAG" and len(results) >= 2):
                confidence = "High"
                confidence_score = min(0.95, round(0.70 + 0.05 * len(results), 2))
            elif top_score > 0.15:
                confidence = "Medium"
                confidence_score = 0.65
            else:
                confidence = "Low"
                confidence_score = 0.40

        # Attempt Ollama generation first
        generator_type = "Deterministic Synthesis"
        generated_text = None

        if self.is_ollama_available():
            generated_text = self.generate_with_ollama(query, results, graph_paths)
            if generated_text:
                generator_type = f"Ollama LLM ({self.model})"

        # Fallback to deterministic synthesis
        if not generated_text:
            generated_text = self.generate_deterministic_synthesis(query, results, graph_paths)

        latency = (time.perf_counter() - start_time) * 1000

        return AnswerResponse(
            query=query,
            answer=generated_text,
            retrieval_method=retrieval_method,
            citations=citations,
            graph_paths=graph_paths,
            retrieved_results=results,
            confidence=confidence,
            confidence_score=confidence_score,
            generator_type=generator_type,
            latency_ms=round(latency, 2)
        )
