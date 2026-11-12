import re
import unicodedata
from typing import Tuple, Dict, Optional
from src.knowledge_graph.schema import NodeType, VALID_NODE_TYPES

# Comprehensive canonical synonym and normalization mapping
SYNONYM_MAP: Dict[str, Tuple[str, str]] = {
    # format: lowercase_alias: (canonical_name, NodeType)
    "co2": ("Carbon Dioxide", NodeType.GREENHOUSE_GAS.value),
    "carbon dioxide": ("Carbon Dioxide", NodeType.GREENHOUSE_GAS.value),
    "ch4": ("Methane", NodeType.GREENHOUSE_GAS.value),
    "methane": ("Methane", NodeType.GREENHOUSE_GAS.value),
    "n2o": ("Nitrous Oxide", NodeType.GREENHOUSE_GAS.value),
    "nitrous oxide": ("Nitrous Oxide", NodeType.GREENHOUSE_GAS.value),
    "greenhouse gases": ("Greenhouse Gases", NodeType.GREENHOUSE_GAS.value),
    "ghgs": ("Greenhouse Gases", NodeType.GREENHOUSE_GAS.value),
    "global warming": ("Global Warming", NodeType.CLIMATE_PHENOMENON.value),
    "climate change": ("Global Warming", NodeType.CLIMATE_PHENOMENON.value),
    "sea level rise": ("Sea Level Rise", NodeType.CLIMATE_PHENOMENON.value),
    "ocean acidification": ("Ocean Acidification", NodeType.CLIMATE_PHENOMENON.value),
    "extreme weather": ("Extreme Weather Events", NodeType.CLIMATE_PHENOMENON.value),
    "extreme weather events": ("Extreme Weather Events", NodeType.CLIMATE_PHENOMENON.value),
    "heatwaves": ("Heatwaves", NodeType.CLIMATE_PHENOMENON.value),
    "heat waves": ("Heatwaves", NodeType.CLIMATE_PHENOMENON.value),
    "monsoon variability": ("Monsoon Variability", NodeType.CLIMATE_PHENOMENON.value),
    "drought": ("Drought", NodeType.CLIMATE_IMPACT.value),
    "droughts": ("Drought", NodeType.CLIMATE_IMPACT.value),
    "agricultural yield decline": ("Agricultural Yield Decline", NodeType.CLIMATE_IMPACT.value),
    "crop failure": ("Agricultural Yield Decline", NodeType.CLIMATE_IMPACT.value),
    "crop yield reduction": ("Agricultural Yield Decline", NodeType.CLIMATE_IMPACT.value),
    "water scarcity": ("Water Scarcity", NodeType.CLIMATE_IMPACT.value),
    "coral bleaching": ("Coral Bleaching", NodeType.CLIMATE_IMPACT.value),
    "coastal flooding": ("Coastal Flooding", NodeType.CLIMATE_IMPACT.value),
    "biodiversity loss": ("Biodiversity Loss", NodeType.CLIMATE_IMPACT.value),
    "india": ("India", NodeType.REGION.value),
    "south asia": ("South Asia", NodeType.REGION.value),
    "arctic": ("Arctic", NodeType.REGION.value),
    "polar regions": ("Arctic", NodeType.REGION.value),
    "high-latitude": ("Arctic", NodeType.REGION.value),
    "amazon": ("Amazon Basin", NodeType.REGION.value),
    "amazon basin": ("Amazon Basin", NodeType.REGION.value),
    "africa": ("Africa", NodeType.REGION.value),
    "europe": ("Europe", NodeType.REGION.value),
    "global": ("Global", NodeType.REGION.value),
    "monsoon": ("Monsoon Variability", NodeType.CLIMATE_PHENOMENON.value),
    "food security": ("Agricultural Yield Decline", NodeType.CLIMATE_IMPACT.value),
    "agriculture": ("Agricultural Yield Decline", NodeType.CLIMATE_IMPACT.value),
    "marine heat": ("Coral Bleaching", NodeType.CLIMATE_IMPACT.value),
    "coral reefs": ("Coral Bleaching", NodeType.CLIMATE_IMPACT.value),
    "carbon pollution": ("Carbon Dioxide", NodeType.GREENHOUSE_GAS.value),
    "clean technology": ("Solar Energy", NodeType.MITIGATION_STRATEGY.value),
    "renewable energy": ("Solar Energy", NodeType.MITIGATION_STRATEGY.value),
    "solar photovoltaics": ("Solar Energy", NodeType.MITIGATION_STRATEGY.value),
    "solar energy": ("Solar Energy", NodeType.MITIGATION_STRATEGY.value),
    "wind power": ("Wind Energy", NodeType.MITIGATION_STRATEGY.value),
    "wind energy": ("Wind Energy", NodeType.MITIGATION_STRATEGY.value),
    "reforestation": ("Afforestation and Reforestation", NodeType.MITIGATION_STRATEGY.value),
    "afforestation": ("Afforestation and Reforestation", NodeType.MITIGATION_STRATEGY.value),
    "carbon capture and storage": ("Carbon Capture and Storage", NodeType.MITIGATION_STRATEGY.value),
    "ccs": ("Carbon Capture and Storage", NodeType.MITIGATION_STRATEGY.value),
    "drip irrigation": ("Drip Irrigation", NodeType.ADAPTATION_STRATEGY.value),
    "micro-irrigation": ("Drip Irrigation", NodeType.ADAPTATION_STRATEGY.value),
    "drought-tolerant crops": ("Drought-Resilient Crop Varieties", NodeType.ADAPTATION_STRATEGY.value),
    "drought-resilient crops": ("Drought-Resilient Crop Varieties", NodeType.ADAPTATION_STRATEGY.value),
    "climate-resilient agriculture": ("Climate-Resilient Agriculture", NodeType.ADAPTATION_STRATEGY.value),
    "rainwater harvesting": ("Rainwater Harvesting", NodeType.ADAPTATION_STRATEGY.value),
    "early warning systems": ("Early Warning Systems", NodeType.ADAPTATION_STRATEGY.value),
    "sea walls": ("Coastal Defenses and Sea Walls", NodeType.ADAPTATION_STRATEGY.value),
    "paris agreement": ("Paris Agreement", NodeType.POLICY.value),
    "kyoto protocol": ("Kyoto Protocol", NodeType.POLICY.value),
    "napcc": ("National Action Plan on Climate Change (NAPCC)", NodeType.POLICY.value),
    "national action plan on climate change": ("National Action Plan on Climate Change (NAPCC)", NodeType.POLICY.value),
    "nationally determined contributions": ("Nationally Determined Contributions (NDCs)", NodeType.POLICY.value),
    "ndcs": ("Nationally Determined Contributions (NDCs)", NodeType.POLICY.value),
    "ipcc": ("Intergovernmental Panel on Climate Change (IPCC)", NodeType.ORGANIZATION.value),
    "intergovernmental panel on climate change": ("Intergovernmental Panel on Climate Change (IPCC)", NodeType.ORGANIZATION.value),
    "unfccc": ("United Nations Framework Convention on Climate Change (UNFCCC)", NodeType.ORGANIZATION.value),
    "wmo": ("World Meteorological Organization (WMO)", NodeType.ORGANIZATION.value),
    "unep": ("United Nations Environment Programme (UNEP)", NodeType.ORGANIZATION.value),
    "icar": ("Indian Council of Agricultural Research (ICAR)", NodeType.ORGANIZATION.value),
    "nasa": ("National Aeronautics and Space Administration (NASA)", NodeType.ORGANIZATION.value),
    "noaa": ("National Oceanic and Atmospheric Administration (NOAA)", NodeType.ORGANIZATION.value),
    "moefcc": ("Ministry of Environment, Forest and Climate Change (MoEFCC)", NodeType.ORGANIZATION.value),
    "nicra": ("National Innovations in Climate Resilient Agriculture (NICRA)", NodeType.POLICY.value),
    "ipcc ar6": ("IPCC Sixth Assessment Report", NodeType.RESEARCH_PAPER.value),
    "ipcc sixth assessment report": ("IPCC Sixth Assessment Report", NodeType.RESEARCH_PAPER.value),
    "emissions gap report": ("UNEP Emissions Gap Report", NodeType.RESEARCH_PAPER.value),
    "ipcc ar5": ("IPCC Fifth Assessment Report", NodeType.RESEARCH_PAPER.value),
    "era5": ("ERA5 Reanalysis Dataset", NodeType.DATASET.value),
    "cmip6": ("CMIP6 Climate Projections", NodeType.DATASET.value),
    "noaa mauna loa": ("NOAA Mauna Loa CO2 Record", NodeType.DATASET.value)
}

def slugify(text: str) -> str:
    """Creates a deterministic slug identifier."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '_', text)

class EntityNormalizer:
    """
    Normalizes extracted entities to canonical forms, assigns stable deterministic IDs,
    and resolves synonyms and aliases.
    """

    @classmethod
    def normalize(cls, raw_name: str, suggested_type: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Normalizes raw entity name.
        Returns: (canonical_name, entity_type, stable_id)
        """
        cleaned = raw_name.strip()
        lower_cleaned = cleaned.lower()

        # Check dictionary mapping
        if lower_cleaned in SYNONYM_MAP:
            canonical, node_type = SYNONYM_MAP[lower_cleaned]
        else:
            canonical = cleaned.title()
            node_type = suggested_type if (suggested_type in VALID_NODE_TYPES) else NodeType.CLIMATE_PHENOMENON.value

        stable_id = f"ent_{slugify(node_type)}_{slugify(canonical)}"
        return canonical, node_type, stable_id

