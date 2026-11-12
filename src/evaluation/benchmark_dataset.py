from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ClimateEvaluationQuery:
    query_id: str
    query: str
    category: str  # "factual", "semantic", "relational", "multi_hop", "adaptation_mitigation"
    expected_answer_facts: List[str]
    expected_relevant_keywords: List[str]
    expected_entity_ids: List[str]
    target_documents: List[str]
    required_focal_entities: List[str]  # Entities that MUST be present to count as relevant
    ground_truth_relevant_count: int = 1  # Known relevant documents/passages in the gold corpus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# 32 Curated, manually verified Climate QA evaluation queries with explicit ground truth
CLIMATE_BENCHMARK_QUERIES: List[ClimateEvaluationQuery] = [
    # Category 1: Simple Factual Questions (1-6)
    ClimateEvaluationQuery(
        query_id="Q01",
        query="What is the primary greenhouse gas driving global warming according to IPCC reports?",
        category="factual",
        expected_answer_facts=["Carbon dioxide (CO2) is the primary greenhouse gas driving radiative forcing and global warming."],
        expected_relevant_keywords=["carbon dioxide", "co2", "radiative forcing", "greenhouse gas"],
        expected_entity_ids=["ent_greenhousegas_carbon_dioxide", "ent_climatephenomenon_global_warming"],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt"],
        required_focal_entities=["ent_greenhousegas_carbon_dioxide"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q02",
        query="What global temperature threshold does the Paris Agreement aim to hold global warming well below?",
        category="factual",
        expected_answer_facts=["The Paris Agreement aims to hold the increase in global average temperature well below 2.0°C and pursue efforts to limit it to 1.5°C."],
        expected_relevant_keywords=["paris agreement", "2.0°c", "1.5°c", "temperature"],
        expected_entity_ids=["ent_policy_paris_agreement", "ent_climatephenomenon_global_warming"],
        target_documents=["unfccc_paris_agreement_official_treaty.txt"],
        required_focal_entities=["ent_policy_paris_agreement"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q03",
        query="What is the role of methane (CH4) as a short-lived climate pollutant and its 20-year warming potential?",
        category="factual",
        expected_answer_facts=["Methane has a 20-year global warming potential over 80 times greater than carbon dioxide."],
        expected_relevant_keywords=["methane", "ch4", "warming potential", "80 times"],
        expected_entity_ids=["ent_greenhousegas_methane", "ent_climatephenomenon_global_warming"],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt", "unep_emissions_gap_report_2023.txt"],
        required_focal_entities=["ent_greenhousegas_methane"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q04",
        query="What atmospheric concentration of carbon dioxide was recorded at NOAA Mauna Loa Observatory?",
        category="factual",
        expected_answer_facts=["Atmospheric carbon dioxide concentrations reached 422.8 parts per million (ppm) at NOAA Mauna Loa in 2024."],
        expected_relevant_keywords=["mauna loa", "422", "carbon dioxide", "ppm"],
        expected_entity_ids=["ent_greenhousegas_carbon_dioxide", "ent_dataset_noaa_mauna_loa_co2_record"],
        target_documents=["nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_greenhousegas_carbon_dioxide"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q05",
        query="What organization coordinates global physical science climate assessments under the United Nations?",
        category="factual",
        expected_answer_facts=["The Intergovernmental Panel on Climate Change (IPCC) established jointly by WMO and UNEP."],
        expected_relevant_keywords=["ipcc", "intergovernmental panel on climate change", "wmo", "unep"],
        expected_entity_ids=["ent_organization_intergovernmental_panel_on_climate_change_ipcc"],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt"],
        required_focal_entities=["ent_organization_intergovernmental_panel_on_climate_change_ipcc"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q06",
        query="What atmospheric phenomenon causes ocean acidification?",
        category="factual",
        expected_answer_facts=["Absorption of excess atmospheric carbon dioxide into seawater forms carbonic acid, lowering ocean pH."],
        expected_relevant_keywords=["ocean acidification", "carbon dioxide", "ph", "seawater"],
        expected_entity_ids=["ent_greenhousegas_carbon_dioxide", "ent_climatephenomenon_ocean_acidification"],
        target_documents=["wmo_state_of_the_global_climate_2023.txt"],
        required_focal_entities=["ent_climatephenomenon_ocean_acidification"],
        ground_truth_relevant_count=1
    ),

    # Category 2: Semantic Retrieval Questions (7-12)
    ClimateEvaluationQuery(
        query_id="Q07",
        query="How does anthropogenic carbon pollution destabilize planetary thermal equilibrium?",
        category="semantic",
        expected_answer_facts=["Emissions of greenhouse gases trap outgoing longwave thermal radiation, creating positive radiative forcing that drives global warming."],
        expected_relevant_keywords=["radiative forcing", "greenhouse gas", "thermal", "carbon dioxide"],
        expected_entity_ids=["ent_greenhousegas_greenhouse_gases", "ent_climatephenomenon_global_warming"],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt", "nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_climatephenomenon_global_warming"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q08",
        query="In what manner do marine heat spikes degrade tropical reef biomes?",
        category="semantic",
        expected_answer_facts=["Marine heatwaves induce thermal stress causing corals to expel symbiotic zooxanthellae algae, triggering severe coral bleaching."],
        expected_relevant_keywords=["marine heatwaves", "coral bleaching", "thermal stress", "reefs"],
        expected_entity_ids=["ent_climatephenomenon_heatwaves", "ent_climateimpact_coral_bleaching"],
        target_documents=["wmo_state_of_the_global_climate_2023.txt"],
        required_focal_entities=["ent_climateimpact_coral_bleaching"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q09",
        query="Why are high-latitude northern polar regions vulnerable to amplified warming?",
        category="semantic",
        expected_answer_facts=["Loss of reflective Arctic sea ice lowers planetary albedo, leading to enhanced solar radiation absorption and Arctic amplification."],
        expected_relevant_keywords=["arctic", "sea ice", "albedo", "arctic amplification"],
        expected_entity_ids=["ent_region_arctic", "ent_climatephenomenon_global_warming"],
        target_documents=["nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_region_arctic"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q10",
        query="How do shifting monsoon precipitation dynamics threaten rural food security?",
        category="semantic",
        expected_answer_facts=["Monsoon variability and erratic rainfall regimes disrupt crop sowing and cause agricultural yield decline in wheat and rice."],
        expected_relevant_keywords=["monsoon variability", "agricultural yield decline", "rice", "wheat"],
        expected_entity_ids=["ent_climatephenomenon_monsoon_variability", "ent_climateimpact_agricultural_yield_decline"],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_climateimpact_agricultural_yield_decline"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q11",
        query="What clean technology approaches help eliminate fossil fuel reliance across power grids?",
        category="semantic",
        expected_answer_facts=["Accelerated deployment of solar photovoltaics and wind energy displacing unabated coal-fired power plants."],
        expected_relevant_keywords=["solar energy", "wind energy", "photovoltaics", "mitigated by"],
        expected_entity_ids=["ent_mitigationstrategy_solar_energy", "ent_mitigationstrategy_wind_energy"],
        target_documents=["unep_emissions_gap_report_2023.txt", "moefcc_national_action_plan_climate_change_india.txt"],
        required_focal_entities=["ent_mitigationstrategy_solar_energy"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q12",
        query="How does tropical deforestation compromise continental hydrological cycles and carbon storage?",
        category="semantic",
        expected_answer_facts=["Deforestation in the Amazon Basin weakens evapotranspiration moisture recycling and converts forest carbon stocks into CO2 emissions."],
        expected_relevant_keywords=["amazon basin", "deforestation", "evapotranspiration", "carbon source"],
        expected_entity_ids=["ent_region_amazon_basin", "ent_greenhousegas_carbon_dioxide"],
        target_documents=["amazon_deforestation_carbon_and_biodiversity_dynamics.txt"],
        required_focal_entities=["ent_region_amazon_basin"],
        ground_truth_relevant_count=1
    ),

    # Category 3: Relationship-Based Questions (13-18)
    ClimateEvaluationQuery(
        query_id="Q13",
        query="What causal relationship exists between greenhouse gas emissions and extreme weather events?",
        category="relational",
        expected_answer_facts=["Greenhouse gases cause global warming, which increases the frequency and intensity of extreme weather events such as heatwaves and droughts."],
        expected_relevant_keywords=["greenhouse gases", "causes", "global warming", "extreme weather events"],
        expected_entity_ids=["ent_greenhousegas_greenhouse_gases", "ent_climatephenomenon_extreme_weather_events"],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt"],
        required_focal_entities=["ent_greenhousegas_greenhouse_gases", "ent_climatephenomenon_extreme_weather_events"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q14",
        query="How does global warming affect water scarcity across South Asia and India?",
        category="relational",
        expected_answer_facts=["Global warming accelerates evapotranspiration and irregular precipitation, exacerbating regional water scarcity and groundwater depletion in India."],
        expected_relevant_keywords=["global warming", "affects", "water scarcity", "india"],
        expected_entity_ids=["ent_climatephenomenon_global_warming", "ent_climateimpact_water_scarcity", "ent_region_india"],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_climateimpact_water_scarcity", "ent_region_india"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q15",
        query="Which international policy treaty recommends Nationally Determined Contributions (NDCs)?",
        category="relational",
        expected_answer_facts=["The Paris Agreement mandates that each Party prepare and communicate successive Nationally Determined Contributions (NDCs)."],
        expected_relevant_keywords=["paris agreement", "nationally determined contributions", "ndcs", "article 4"],
        expected_entity_ids=["ent_policy_paris_agreement", "ent_policy_nationally_determined_contributions_ndcs"],
        target_documents=["unfccc_paris_agreement_official_treaty.txt"],
        required_focal_entities=["ent_policy_paris_agreement", "ent_policy_nationally_determined_contributions_ndcs"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q16",
        query="What climate phenomenon causes sea level rise according to satellite and ocean measurements?",
        category="relational",
        expected_answer_facts=["Global warming causes thermal expansion of ocean water and the melting of mountain glaciers and ice sheets, driving sea level rise."],
        expected_relevant_keywords=["global warming", "causes", "sea level rise", "thermal expansion"],
        expected_entity_ids=["ent_climatephenomenon_global_warming", "ent_climatephenomenon_sea_level_rise"],
        target_documents=["nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_climatephenomenon_global_warming", "ent_climatephenomenon_sea_level_rise"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q17",
        query="How is agricultural yield decline in India adapted through water conservation techniques?",
        category="relational",
        expected_answer_facts=["Agricultural yield decline in India is adapted through drip irrigation, cutting water use by 40 to 60 percent."],
        expected_relevant_keywords=["agricultural yield decline", "adapted by", "drip irrigation", "india"],
        expected_entity_ids=["ent_climateimpact_agricultural_yield_decline", "ent_adaptationstrategy_drip_irrigation", "ent_region_india"],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_climateimpact_agricultural_yield_decline", "ent_adaptationstrategy_drip_irrigation"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q18",
        query="What climate impact affects coral reefs in tropical oceans due to marine heatwaves?",
        category="relational",
        expected_answer_facts=["Marine heatwaves and ocean acidification affect tropical reefs by causing widespread coral bleaching."],
        expected_relevant_keywords=["affects", "coral bleaching", "heatwaves", "ocean acidification"],
        expected_entity_ids=["ent_climatephenomenon_heatwaves", "ent_climateimpact_coral_bleaching"],
        target_documents=["wmo_state_of_the_global_climate_2023.txt"],
        required_focal_entities=["ent_climateimpact_coral_bleaching"],
        ground_truth_relevant_count=1
    ),

    # Category 4: Multi-Hop Questions (19-24)
    ClimateEvaluationQuery(
        query_id="Q19",
        query="How does carbon dioxide emission lead to agricultural yield decline in India, and what adaptation strategy is documented?",
        category="multi_hop",
        expected_answer_facts=["CO2 causes Global Warming -> induces Heatwaves & Drought -> causes Agricultural Yield Decline in India -> adapted by Drought-Resilient Crop Varieties and Drip Irrigation."],
        expected_relevant_keywords=["carbon dioxide", "global warming", "agricultural yield decline", "india", "drip irrigation", "drought-resilient"],
        expected_entity_ids=[
            "ent_greenhousegas_carbon_dioxide",
            "ent_climatephenomenon_global_warming",
            "ent_climateimpact_agricultural_yield_decline",
            "ent_region_india",
            "ent_adaptationstrategy_drip_irrigation"
        ],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt", "icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_climateimpact_agricultural_yield_decline", "ent_region_india", "ent_adaptationstrategy_drip_irrigation"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q20",
        query="Trace how industrial greenhouse gases cause coastal flooding, and identify recommended protective infrastructure.",
        category="multi_hop",
        expected_answer_facts=["Greenhouse gases cause Global Warming -> Sea Level Rise -> Coastal Flooding -> adapted through Coastal Defenses and Sea Walls."],
        expected_relevant_keywords=["greenhouse gases", "sea level rise", "coastal flooding", "sea walls"],
        expected_entity_ids=[
            "ent_greenhousegas_greenhouse_gases",
            "ent_climatephenomenon_sea_level_rise",
            "ent_climateimpact_coastal_flooding",
            "ent_adaptationstrategy_coastal_defenses_and_sea_walls"
        ],
        target_documents=["nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_climatephenomenon_sea_level_rise", "ent_climateimpact_coastal_flooding"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q21",
        query="How does fossil fuel burning trigger water scarcity in India, and how can farming communities adapt?",
        category="multi_hop",
        expected_answer_facts=["Fossil fuel CO2 -> Global Warming -> Drought and Groundwater Depletion -> Water Scarcity in India -> adapted through Rainwater Harvesting."],
        expected_relevant_keywords=["carbon dioxide", "water scarcity", "india", "rainwater harvesting"],
        expected_entity_ids=[
            "ent_greenhousegas_carbon_dioxide",
            "ent_climateimpact_water_scarcity",
            "ent_region_india",
            "ent_adaptationstrategy_rainwater_harvesting"
        ],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_climateimpact_water_scarcity", "ent_region_india", "ent_adaptationstrategy_rainwater_harvesting"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q22",
        query="Explain the link connecting methane emissions to extreme heatwaves and international synthesis by the IPCC.",
        category="multi_hop",
        expected_answer_facts=["Methane causes Global Warming -> intensifies Heatwaves -> evaluated by the Intergovernmental Panel on Climate Change (IPCC)."],
        expected_relevant_keywords=["methane", "heatwaves", "ipcc", "intergovernmental panel on climate change"],
        expected_entity_ids=[
            "ent_greenhousegas_methane",
            "ent_climatephenomenon_heatwaves",
            "ent_organization_intergovernmental_panel_on_climate_change_ipcc"
        ],
        target_documents=["ipcc_ar6_wg1_summary_for_policymakers.txt"],
        required_focal_entities=["ent_greenhousegas_methane", "ent_climatephenomenon_heatwaves", "ent_organization_intergovernmental_panel_on_climate_change_ipcc"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q23",
        query="Trace how Amazon deforestation intensifies global warming and subsequently affects global biodiversity loss.",
        category="multi_hop",
        expected_answer_facts=["Amazon Deforestation releases CO2 -> accelerates Global Warming -> destroys habitats driving Biodiversity Loss."],
        expected_relevant_keywords=["amazon basin", "deforestation", "carbon dioxide", "biodiversity loss"],
        expected_entity_ids=[
            "ent_region_amazon_basin",
            "ent_greenhousegas_carbon_dioxide",
            "ent_climateimpact_biodiversity_loss"
        ],
        target_documents=["amazon_deforestation_carbon_and_biodiversity_dynamics.txt"],
        required_focal_entities=["ent_region_amazon_basin", "ent_climateimpact_biodiversity_loss"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q24",
        query="How does global warming affect agriculture in India, and what adaptation strategies are documented?",
        category="multi_hop",
        expected_answer_facts=["Global Warming causes Heatwaves affecting crops in India, which is adapted through Drip Irrigation and Climate-Resilient Agriculture."],
        expected_relevant_keywords=["global warming", "agriculture", "india", "drip irrigation", "climate-resilient agriculture"],
        expected_entity_ids=[
            "ent_climatephenomenon_global_warming",
            "ent_region_india",
            "ent_climateimpact_agricultural_yield_decline",
            "ent_adaptationstrategy_drip_irrigation"
        ],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_region_india", "ent_climateimpact_agricultural_yield_decline", "ent_adaptationstrategy_drip_irrigation"],
        ground_truth_relevant_count=1
    ),

    # Category 5: Adaptation & Mitigation Strategy Questions (25-32)
    ClimateEvaluationQuery(
        query_id="Q25",
        query="What adaptation strategies does the Indian Council of Agricultural Research (ICAR) recommend for drought resilience?",
        category="adaptation_mitigation",
        expected_answer_facts=["ICAR recommends drought-resilient crop varieties (CR Dhan 801, DRR 42), drip irrigation, and rainwater harvesting farm ponds."],
        expected_relevant_keywords=["icar", "drought-resilient crop varieties", "drip irrigation", "farm ponds"],
        expected_entity_ids=[
            "ent_organization_indian_council_of_agricultural_research_icar",
            "ent_adaptationstrategy_drought_resilient_crop_varieties"
        ],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_organization_indian_council_of_agricultural_research_icar", "ent_adaptationstrategy_drought_resilient_crop_varieties"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q26",
        query="How do solar energy and wind power serve as primary mitigation strategies according to UNEP?",
        category="adaptation_mitigation",
        expected_answer_facts=["Solar photovoltaics and wind energy displace unabated coal power, directly eliminating carbon dioxide emissions."],
        expected_relevant_keywords=["solar energy", "wind energy", "mitigation", "unep", "photovoltaics"],
        expected_entity_ids=[
            "ent_mitigationstrategy_solar_energy",
            "ent_mitigationstrategy_wind_energy",
            "ent_organization_united_nations_environment_programme_unep"
        ],
        target_documents=["unep_emissions_gap_report_2023.txt"],
        required_focal_entities=["ent_mitigationstrategy_solar_energy", "ent_mitigationstrategy_wind_energy"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q27",
        query="What is the role of afforestation and reforestation in biological carbon sequestration?",
        category="adaptation_mitigation",
        expected_answer_facts=["Afforestation and reforestation sequester atmospheric carbon dioxide into terrestrial forest biomass and soil carbon stocks."],
        expected_relevant_keywords=["afforestation and reforestation", "carbon sink", "mitigation", "green india"],
        expected_entity_ids=[
            "ent_mitigationstrategy_afforestation_and_reforestation",
            "ent_greenhousegas_carbon_dioxide"
        ],
        target_documents=["moefcc_national_action_plan_climate_change_india.txt", "amazon_deforestation_carbon_and_biodiversity_dynamics.txt"],
        required_focal_entities=["ent_mitigationstrategy_afforestation_and_reforestation"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q28",
        query="How does rainwater harvesting assist climate adaptation in drought-prone agrarian regions?",
        category="adaptation_mitigation",
        expected_answer_facts=["Rainwater harvesting structures capture monsoon runoff, recharge groundwater aquifers, and buffer against drought spells."],
        expected_relevant_keywords=["rainwater harvesting", "groundwater", "drought", "farm ponds"],
        expected_entity_ids=[
            "ent_adaptationstrategy_rainwater_harvesting",
            "ent_climateimpact_drought"
        ],
        target_documents=["icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_adaptationstrategy_rainwater_harvesting"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q29",
        query="What early warning systems are implemented to reduce vulnerability to extreme heatwaves?",
        category="adaptation_mitigation",
        expected_answer_facts=["Meteorological early warning systems and agro-advisories alert populations and farmers, reducing disaster asset losses by 30 percent."],
        expected_relevant_keywords=["early warning systems", "heatwaves", "adaptation", "wmo"],
        expected_entity_ids=[
            "ent_adaptationstrategy_early_warning_systems",
            "ent_climatephenomenon_heatwaves"
        ],
        target_documents=["wmo_state_of_the_global_climate_2023.txt", "icar_nicra_climate_adaptation_indian_agriculture.txt"],
        required_focal_entities=["ent_adaptationstrategy_early_warning_systems"],
        ground_truth_relevant_count=2
    ),
    ClimateEvaluationQuery(
        query_id="Q30",
        query="How does the National Action Plan on Climate Change (NAPCC) support adaptation and mitigation in India?",
        category="adaptation_mitigation",
        expected_answer_facts=["NAPCC establishes eight national missions encompassing Solar Energy, Water Conservation, Sustainable Agriculture, and Green India."],
        expected_relevant_keywords=["napcc", "national action plan on climate change", "solar mission", "water mission"],
        expected_entity_ids=[
            "ent_policy_national_action_plan_on_climate_change_napcc"
        ],
        target_documents=["moefcc_national_action_plan_climate_change_india.txt"],
        required_focal_entities=["ent_policy_national_action_plan_on_climate_change_napcc"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q31",
        query="What coastal defenses can protect vulnerable low-lying regions against sea level rise?",
        category="adaptation_mitigation",
        expected_answer_facts=["Engineered sea walls, tidal flood barriers, and coastal defenses mitigate coastal flooding and shoreline erosion."],
        expected_relevant_keywords=["coastal defenses and sea walls", "sea level rise", "coastal flooding"],
        expected_entity_ids=[
            "ent_adaptationstrategy_coastal_defenses_and_sea_walls",
            "ent_climatephenomenon_sea_level_rise"
        ],
        target_documents=["nasa_noaa_climate_vitals_and_carbon_cycle.txt"],
        required_focal_entities=["ent_adaptationstrategy_coastal_defenses_and_sea_walls"],
        ground_truth_relevant_count=1
    ),
    ClimateEvaluationQuery(
        query_id="Q32",
        query="What is the role of Carbon Capture and Storage (CCS) in industrial mitigation?",
        category="adaptation_mitigation",
        expected_answer_facts=["Carbon capture and storage (CCS) captures CO2 emissions from hard-to-abate industrial point sources and stores it in geological formations."],
        expected_relevant_keywords=["carbon capture and storage", "ccs", "industrial", "mitigation"],
        expected_entity_ids=[
            "ent_mitigationstrategy_carbon_capture_and_storage",
            "ent_greenhousegas_carbon_dioxide"
        ],
        target_documents=["unep_emissions_gap_report_2023.txt"],
        required_focal_entities=["ent_mitigationstrategy_carbon_capture_and_storage"],
        ground_truth_relevant_count=1
    )
]
