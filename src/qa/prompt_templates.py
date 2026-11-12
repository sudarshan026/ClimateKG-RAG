def build_grounded_qa_prompt(query: str, passages: list[dict], graph_paths: list[str]) -> str:
    """
    Constructs a scientifically grounded QA prompt for LLM generation.
    Strictly prevents hallucination and enforces exact citations.
    """
    context_blocks = []
    for idx, p in enumerate(passages, start=1):
        source_label = f"[{idx}] Source: {p.get('filename', 'Unknown')} (Page {p.get('page_number', 1)}, Chunk: {p.get('chunk_id', '')})"
        context_blocks.append(f"{source_label}\n{p.get('text', '').strip()}")

    formatted_context = "\n\n".join(context_blocks)
    formatted_paths = "\n".join(f"- {path}" for path in graph_paths) if graph_paths else "None identified."

    prompt = f"""You are an expert scientific climate AI assistant for ClimateKG-RAG (UN SDG 13: Climate Action).
Your objective is to answer the user's climate question using ONLY the provided verified document excerpts and Knowledge Graph relational paths.

CRITICAL GROUNDING RULES:
1. Ground every claim directly in the provided evidence. Cite sources using square bracket notation, e.g., [Source: filename, Page X].
2. If multi-hop relational paths from the Knowledge Graph are provided, explicitly mention how the entities connect (e.g., "According to the Knowledge Graph: [Entity A] causes [Entity B] which affects [Entity C]").
3. NEVER fabricate or extrapolate facts, citations, dates, or statistics not supported by the evidence.
4. If the retrieved passages and graph paths do not contain sufficient evidence to answer the question thoroughly, explicitly state: "Insufficient evidence in the ingested documents to fully answer this question."

=== KNOWLEDGE GRAPH RELATIONAL PATHS ===
{formatted_paths}

=== RETRIEVED DOCUMENT EVIDENCE ===
{formatted_context}

=== USER QUESTION ===
{query}

=== SCIENTIFIC GROUNDED ANSWER ===
"""
    return prompt

