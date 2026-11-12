import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import requests

from config.settings import settings
from src.knowledge_graph.schema import (
    Entity, Relation, NodeType, RelationType,
    VALID_NODE_TYPES, VALID_RELATION_TYPES
)
from src.knowledge_graph.normalizer import EntityNormalizer, SYNONYM_MAP, slugify
from src.chunking.chunker import DocumentChunk

logger = logging.getLogger(__name__)

# Climate Domain Keyword Lexicon for Fallback Matching
CLIMATE_LEXICON: Dict[str, str] = {
    # Greenhouse Gases
    "carbon dioxide": NodeType.GREENHOUSE_GAS.value,
    "co2": NodeType.GREENHOUSE_GAS.value,
    "methane": NodeType.GREENHOUSE_GAS.value,
    "ch4": NodeType.GREENHOUSE_GAS.value,
    "nitrous oxide": NodeType.GREENHOUSE_GAS.value,
    "n2o": NodeType.GREENHOUSE_GAS.value,
    "greenhouse gases": NodeType.GREENHOUSE_GAS.value,
    "ghgs": NodeType.GREENHOUSE_GAS.value,

    # Climate Phenomena
    "global warming": NodeType.CLIMATE_PHENOMENON.value,
    "climate change": NodeType.CLIMATE_PHENOMENON.value,
    "sea level rise": NodeType.CLIMATE_PHENOMENON.value,
    "ocean acidification": NodeType.CLIMATE_PHENOMENON.value,
    "extreme weather": NodeType.CLIMATE_PHENOMENON.value,
    "heatwaves": NodeType.CLIMATE_PHENOMENON.value,
    "monsoon variability": NodeType.CLIMATE_PHENOMENON.value,
    "temperature anomalies": NodeType.CLIMATE_PHENOMENON.value,

    # Climate Impacts
    "drought": NodeType.CLIMATE_IMPACT.value,
    "droughts": NodeType.CLIMATE_IMPACT.value,
    "agricultural yield decline": NodeType.CLIMATE_IMPACT.value,
    "crop yield reduction": NodeType.CLIMATE_IMPACT.value,
    "crop failure": NodeType.CLIMATE_IMPACT.value,
    "water scarcity": NodeType.CLIMATE_IMPACT.value,
    "coral bleaching": NodeType.CLIMATE_IMPACT.value,
    "coastal flooding": NodeType.CLIMATE_IMPACT.value,
    "biodiversity loss": NodeType.CLIMATE_IMPACT.value,
    "forest fires": NodeType.CLIMATE_IMPACT.value,

    # Regions
    "south asia": NodeType.REGION.value,
    "india": NodeType.REGION.value,
    "arctic": NodeType.REGION.value,
    "amazon basin": NodeType.REGION.value,
    "africa": NodeType.REGION.value,
    "europe": NodeType.REGION.value,
    "global": NodeType.REGION.value,

    # Mitigation Strategies
    "solar energy": NodeType.MITIGATION_STRATEGY.value,
    "solar photovoltaics": NodeType.MITIGATION_STRATEGY.value,
    "wind energy": NodeType.MITIGATION_STRATEGY.value,
    "afforestation and reforestation": NodeType.MITIGATION_STRATEGY.value,
    "afforestation": NodeType.MITIGATION_STRATEGY.value,
    "reforestation": NodeType.MITIGATION_STRATEGY.value,
    "carbon capture and storage": NodeType.MITIGATION_STRATEGY.value,
    "renewable energy transition": NodeType.MITIGATION_STRATEGY.value,

    # Adaptation Strategies
    "drip irrigation": NodeType.ADAPTATION_STRATEGY.value,
    "drought-resilient crop varieties": NodeType.ADAPTATION_STRATEGY.value,
    "climate-resilient agriculture": NodeType.ADAPTATION_STRATEGY.value,
    "rainwater harvesting": NodeType.ADAPTATION_STRATEGY.value,
    "early warning systems": NodeType.ADAPTATION_STRATEGY.value,
    "sea walls": NodeType.ADAPTATION_STRATEGY.value,

    # Policies
    "paris agreement": NodeType.POLICY.value,
    "kyoto protocol": NodeType.POLICY.value,
    "national action plan on climate change (napcc)": NodeType.POLICY.value,
    "nationally determined contributions (ndcs)": NodeType.POLICY.value,

    # Organizations
    "intergovernmental panel on climate change (ipcc)": NodeType.ORGANIZATION.value,
    "ipcc": NodeType.ORGANIZATION.value,
    "united nations framework convention on climate change (unfccc)": NodeType.ORGANIZATION.value,
    "unfccc": NodeType.ORGANIZATION.value,
    "world meteorological organization (wmo)": NodeType.ORGANIZATION.value,
    "united nations environment programme (unep)": NodeType.ORGANIZATION.value,
    "indian council of agricultural research (icar)": NodeType.ORGANIZATION.value,
    "national aeronautics and space administration (nasa)": NodeType.ORGANIZATION.value,
    "nasa": NodeType.ORGANIZATION.value,
    "national oceanic and atmospheric administration (noaa)": NodeType.ORGANIZATION.value,
    "noaa": NodeType.ORGANIZATION.value,
    "moefcc": NodeType.ORGANIZATION.value,
    "national innovations in climate resilient agriculture (nicra)": NodeType.POLICY.value,
    "nicra": NodeType.POLICY.value,

    # Research Papers & Datasets
    "ipcc sixth assessment report": NodeType.RESEARCH_PAPER.value,
    "ipcc ar6": NodeType.RESEARCH_PAPER.value,
    "era5 reanalysis dataset": NodeType.DATASET.value,
    "cmip6 climate projections": NodeType.DATASET.value
}

# Relation Pattern Regexes for Deterministic Extraction
RELATION_PATTERNS = [
    # CAUSES
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:causes|caused|causing|leads to|leading to|results in|resulting in|drives|driving|induces|inducing)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.CAUSES.value
    ),
    # AFFECTS
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:affects|affecting|impacts|impacting|threatens|threatening|disrupts|disrupting)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.AFFECTS.value
    ),
    # OCCURS_IN
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:occurs in|occurring in|observed in|documented in|recorded across|in the region of)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.OCCURS_IN.value
    ),
    # MITIGATED_BY
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:can be mitigated by|is mitigated through|is mitigated by|reduced through|countered by)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.MITIGATED_BY.value
    ),
    # ADAPTED_BY
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:adapted through|adapted by|managed via|managed through|coped with through)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.ADAPTED_BY.value
    ),
    # RECOMMENDED_BY
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:is recommended by|recommended by|endorsed by|mandated by)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.RECOMMENDED_BY.value
    ),
    # SUPPORTED_BY
    (
        re.compile(r'(\b[A-Za-z0-9_\-\s]{3,35}\b)\s+(?:supported by|evidenced by|projected by data from|derived from)\s+(\b[A-Za-z0-9_\-\s]{3,35}\b)', re.IGNORECASE),
        RelationType.SUPPORTED_BY.value
    )
]

class KnowledgeGraphExtractor:
    """
    Extracts structured entities and evidence-linked relationships from text chunks.
    Prefers Ollama LLM structured extraction when available, falling back to
    deterministic rule-based pattern matching with exact provenance guarantees.
    """

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None):
        self.ollama_url = ollama_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self._ollama_checked = False
        self._ollama_available = False

    def is_ollama_available(self) -> bool:
        """Checks if local Ollama daemon is reachable and responding."""
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

    def extract_with_ollama(self, chunk: DocumentChunk) -> Optional[Tuple[List[Entity], List[Relation]]]:
        """Performs structured JSON entity and relationship extraction using Ollama."""
        prompt = f"""You are a scientific Climate Knowledge Graph extraction expert.
Extract entities and relationships from the provided climate text passage.

Permitted Entity Types:
- ClimatePhenomenon (e.g. Global Warming, Sea Level Rise, Heatwaves, Monsoon Variability)
- GreenhouseGas (e.g. Carbon Dioxide, Methane, Nitrous Oxide)
- Region (e.g. South Asia, India, Arctic, Amazon Basin)
- ClimateImpact (e.g. Agricultural Yield Decline, Water Scarcity, Coral Bleaching, Drought)
- MitigationStrategy (e.g. Solar Energy, Wind Energy, Afforestation, Carbon Capture)
- AdaptationStrategy (e.g. Drip Irrigation, Drought-Resilient Crop Varieties, Sea Walls)
- Policy (e.g. Paris Agreement, NAPCC)
- ResearchPaper (e.g. IPCC Sixth Assessment Report)
- Organization (e.g. IPCC, UNFCCC, ICAR)
- Dataset (e.g. ERA5, CMIP6)

Permitted Relationship Types:
- CAUSES, AFFECTS, OCCURS_IN, MITIGATED_BY, ADAPTED_BY, RECOMMENDED_BY, SUPPORTED_BY, RELATED_TO

Text to analyze:
\"\"\"{chunk.text}\"\"\"

Respond STRICTLY with valid JSON without any markdown formatting or explanations:
{{
  "entities": [
    {{"name": "Entity Name", "type": "EntityType"}}
  ],
  "relationships": [
    {{"source": "Source Name", "target": "Target Name", "type": "RelationType", "evidence": "Exact excerpt from text"}}
  ]
}}
"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=settings.OLLAMA_TIMEOUT
            )
            if resp.status_code != 200:
                return None

            data = resp.json().get("response", "")
            parsed = json.loads(data)

            extracted_entities: List[Entity] = []
            extracted_relations: List[Relation] = []
            seen_entity_ids = set()

            for ent_dict in parsed.get("entities", []):
                name = ent_dict.get("name", "").strip()
                etype = ent_dict.get("type", "").strip()
                if not name or etype not in VALID_NODE_TYPES:
                    continue
                c_name, valid_type, s_id = EntityNormalizer.normalize(name, etype)
                if s_id not in seen_entity_ids:
                    seen_entity_ids.add(s_id)
                    extracted_entities.append(Entity(
                        id=s_id,
                        name=c_name,
                        type=valid_type,
                        canonical_name=c_name,
                        properties={"source_chunk": chunk.chunk_id, "extraction_method": "ollama_llm"}
                    ))

            for rel_dict in parsed.get("relationships", []):
                src = rel_dict.get("source", "").strip()
                tgt = rel_dict.get("target", "").strip()
                rtype = rel_dict.get("type", "").strip()
                ev = rel_dict.get("evidence", "").strip() or chunk.text[:200]

                if not src or not tgt or rtype not in VALID_RELATION_TYPES:
                    continue

                c_src, src_type, s_id = EntityNormalizer.normalize(src)
                c_tgt, tgt_type, t_id = EntityNormalizer.normalize(tgt)

                # Ensure source and target entities exist
                if s_id not in seen_entity_ids:
                    seen_entity_ids.add(s_id)
                    extracted_entities.append(Entity(id=s_id, name=c_src, type=src_type, canonical_name=c_src))
                if t_id not in seen_entity_ids:
                    seen_entity_ids.add(t_id)
                    extracted_entities.append(Entity(id=t_id, name=c_tgt, type=tgt_type, canonical_name=c_tgt))

                rel_id = f"rel_{s_id}_{rtype}_{t_id}_{chunk.chunk_id}"
                extracted_relations.append(Relation(
                    id=rel_id,
                    source_id=s_id,
                    source_name=c_src,
                    target_id=t_id,
                    target_name=c_tgt,
                    type=rtype,
                    evidence_text=ev,
                    source_chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    confidence=0.95,
                    is_inferred=False,
                    properties={"extraction_method": "ollama_llm"}
                ))

            return extracted_entities, extracted_relations

        except Exception as e:
            logger.warning(f"Ollama extraction failed, falling back to deterministic extraction: {e}")
            return None

    def extract_deterministic(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relation]]:
        """
        Deterministic, rule-based entity and relationship extractor.
        Guarantees exact evidence excerpts and 100% stable provenance without relying on external APIs.
        """
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []
        text_lower = chunk.text.lower()

        # Step 1: Lexicon-based Entity Recognition
        for term, etype in CLIMATE_LEXICON.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                c_name, norm_type, stable_id = EntityNormalizer.normalize(term, etype)
                if stable_id not in entities:
                    entities[stable_id] = Entity(
                        id=stable_id,
                        name=c_name,
                        type=norm_type,
                        canonical_name=c_name,
                        properties={"source_chunk": chunk.chunk_id, "extraction_method": "deterministic_lexicon"}
                    )

        # Step 2: Sentence-level Relationship Extraction
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        for sentence in sentences:
            s_clean = sentence.strip()
            if not s_clean:
                continue

            for pattern, rel_type in RELATION_PATTERNS:
                matches = pattern.finditer(s_clean)
                for match in matches:
                    raw_src = match.group(1).strip()
                    raw_tgt = match.group(2).strip()

                    # Find best matches among identified entities or resolve if close
                    src_ent = self._find_matching_entity(raw_src, entities)
                    tgt_ent = self._find_matching_entity(raw_tgt, entities)

                    if src_ent and tgt_ent and src_ent.id != tgt_ent.id:
                        rel_id = f"rel_{src_ent.id}_{rel_type}_{tgt_ent.id}_{chunk.chunk_id}"
                        # Check if duplicate relation already recorded
                        if not any(r.id == rel_id for r in relations):
                            relations.append(Relation(
                                id=rel_id,
                                source_id=src_ent.id,
                                source_name=src_ent.name,
                                target_id=tgt_ent.id,
                                target_name=tgt_ent.name,
                                type=rel_type,
                                evidence_text=s_clean,
                                source_chunk_id=chunk.chunk_id,
                                doc_id=chunk.doc_id,
                                filename=chunk.filename,
                                page_number=chunk.page_number,
                                confidence=0.88,
                                is_inferred=False,
                                properties={"extraction_method": "deterministic_pattern"}
                            ))

            # Step 3: Sentence Co-occurrence fallback for verified domain pairs
            # e.g., If a sentence mentions both a Phenomenon/Impact and a Region
            # or Phenomenon and Strategy, create verified relation with sentence evidence
            ents_in_sentence = [ent for ent in entities.values() if ent.canonical_name.lower() in s_clean.lower()]
            if len(ents_in_sentence) >= 2:
                for i in range(len(ents_in_sentence)):
                    for j in range(i + 1, len(ents_in_sentence)):
                        e1, e2 = ents_in_sentence[i], ents_in_sentence[j]
                        rel_type, src, tgt = self._infer_rel_type(e1, e2, s_clean)
                        if rel_type:
                            rel_id = f"rel_{src.id}_{rel_type}_{tgt.id}_{chunk.chunk_id}"
                            if not any(r.id == rel_id for r in relations):
                                relations.append(Relation(
                                    id=rel_id,
                                    source_id=src.id,
                                    source_name=src.name,
                                    target_id=tgt.id,
                                    target_name=tgt.name,
                                    type=rel_type,
                                    evidence_text=s_clean,
                                    source_chunk_id=chunk.chunk_id,
                                    doc_id=chunk.doc_id,
                                    filename=chunk.filename,
                                    page_number=chunk.page_number,
                                    confidence=0.82,
                                    is_inferred=False,
                                    properties={"extraction_method": "sentence_cooccurrence"}
                                ))

        return list(entities.values()), relations

    @staticmethod
    def _find_matching_entity(phrase: str, entities: Dict[str, Entity]) -> Optional[Entity]:
        p_low = phrase.lower()
        for ent in entities.values():
            if ent.canonical_name.lower() in p_low or p_low in ent.canonical_name.lower():
                return ent
        return None

    @staticmethod
    def _infer_rel_type(e1: Entity, e2: Entity, sentence: str) -> Tuple[Optional[str], Entity, Entity]:
        """Determines semantic relation type based on entity types and sentence content."""
        s_low = sentence.lower()
        t1, t2 = e1.type, e2.type

        # GreenhouseGas -> CAUSES -> ClimatePhenomenon
        if t1 == NodeType.GREENHOUSE_GAS.value and t2 == NodeType.CLIMATE_PHENOMENON.value:
            return RelationType.CAUSES.value, e1, e2
        if t2 == NodeType.GREENHOUSE_GAS.value and t1 == NodeType.CLIMATE_PHENOMENON.value:
            return RelationType.CAUSES.value, e2, e1

        # ClimatePhenomenon -> CAUSES / AFFECTS -> ClimateImpact
        if t1 == NodeType.CLIMATE_PHENOMENON.value and t2 == NodeType.CLIMATE_IMPACT.value:
            return RelationType.AFFECTS.value, e1, e2
        if t2 == NodeType.CLIMATE_PHENOMENON.value and t1 == NodeType.CLIMATE_IMPACT.value:
            return RelationType.AFFECTS.value, e2, e1

        # ClimateImpact / ClimatePhenomenon -> OCCURS_IN -> Region
        if t1 in (NodeType.CLIMATE_IMPACT.value, NodeType.CLIMATE_PHENOMENON.value) and t2 == NodeType.REGION.value:
            return RelationType.OCCURS_IN.value, e1, e2
        if t2 in (NodeType.CLIMATE_IMPACT.value, NodeType.CLIMATE_PHENOMENON.value) and t1 == NodeType.REGION.value:
            return RelationType.OCCURS_IN.value, e2, e1

        # ClimateImpact -> ADAPTED_BY -> AdaptationStrategy
        if t1 == NodeType.CLIMATE_IMPACT.value and t2 == NodeType.ADAPTATION_STRATEGY.value:
            return RelationType.ADAPTED_BY.value, e1, e2
        if t2 == NodeType.CLIMATE_IMPACT.value and t1 == NodeType.ADAPTATION_STRATEGY.value:
            return RelationType.ADAPTED_BY.value, e2, e1

        # ClimatePhenomenon / GreenhouseGas -> MITIGATED_BY -> MitigationStrategy
        if t1 in (NodeType.CLIMATE_PHENOMENON.value, NodeType.GREENHOUSE_GAS.value) and t2 == NodeType.MITIGATION_STRATEGY.value:
            return RelationType.MITIGATED_BY.value, e1, e2
        if t2 in (NodeType.CLIMATE_PHENOMENON.value, NodeType.GREENHOUSE_GAS.value) and t1 == NodeType.MITIGATION_STRATEGY.value:
            return RelationType.MITIGATED_BY.value, e2, e1

        # Strategy -> RECOMMENDED_BY -> Policy / Organization
        if t1 in (NodeType.MITIGATION_STRATEGY.value, NodeType.ADAPTATION_STRATEGY.value) and t2 in (NodeType.POLICY.value, NodeType.ORGANIZATION.value):
            return RelationType.RECOMMENDED_BY.value, e1, e2
        if t2 in (NodeType.MITIGATION_STRATEGY.value, NodeType.ADAPTATION_STRATEGY.value) and t1 in (NodeType.POLICY.value, NodeType.ORGANIZATION.value):
            return RelationType.RECOMMENDED_BY.value, e2, e1

        # Policy / Claim -> SUPPORTED_BY -> ResearchPaper / Dataset
        if t1 in (NodeType.POLICY.value, NodeType.CLIMATE_PHENOMENON.value) and t2 in (NodeType.RESEARCH_PAPER.value, NodeType.DATASET.value):
            return RelationType.SUPPORTED_BY.value, e1, e2
        if t2 in (NodeType.POLICY.value, NodeType.CLIMATE_PHENOMENON.value) and t1 in (NodeType.RESEARCH_PAPER.value, NodeType.DATASET.value):
            return RelationType.SUPPORTED_BY.value, e2, e1

        return None, e1, e2

    def extract(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relation]]:
        """
        Unified extraction entry point.
        Attempts Ollama LLM extraction if enabled and online; falls back cleanly to deterministic extraction.
        """
        if self.is_ollama_available():
            logger.info(f"Extracting with Ollama LLM for chunk {chunk.chunk_id}")
            result = self.extract_with_ollama(chunk)
            if result:
                return result

        logger.debug(f"Using deterministic extraction for chunk {chunk.chunk_id}")
        return self.extract_deterministic(chunk)
