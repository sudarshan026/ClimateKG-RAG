from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

class NodeType(str, Enum):
    CLIMATE_PHENOMENON = "ClimatePhenomenon"
    GREENHOUSE_GAS = "GreenhouseGas"
    REGION = "Region"
    CLIMATE_IMPACT = "ClimateImpact"
    MITIGATION_STRATEGY = "MitigationStrategy"
    ADAPTATION_STRATEGY = "AdaptationStrategy"
    POLICY = "Policy"
    RESEARCH_PAPER = "ResearchPaper"
    ORGANIZATION = "Organization"
    DATASET = "Dataset"

class RelationType(str, Enum):
    CAUSES = "CAUSES"
    AFFECTS = "AFFECTS"
    OCCURS_IN = "OCCURS_IN"
    MITIGATED_BY = "MITIGATED_BY"
    ADAPTED_BY = "ADAPTED_BY"
    RECOMMENDED_BY = "RECOMMENDED_BY"
    SUPPORTED_BY = "SUPPORTED_BY"
    RELATED_TO = "RELATED_TO"

VALID_NODE_TYPES = {t.value for t in NodeType}
VALID_RELATION_TYPES = {t.value for t in RelationType}

@dataclass
class Entity:
    id: str
    name: str
    type: str  # Must be one of NodeType
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(**data)

@dataclass
class Relation:
    id: str
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    type: str  # Must be one of RelationType
    evidence_text: str  # Mandatory exact excerpt from text
    source_chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    confidence: float = 1.0
    is_inferred: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(**data)

