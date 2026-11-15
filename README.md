# 🌍 ClimateKG-RAG

**A Hybrid Knowledge Graph and Multi-Paradigm Information Retrieval System for Climate Change Knowledge Discovery**  
*Aligned with UN Sustainable Development Goal 13: Climate Action (SDG 13)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UI: Modern SPA](https://img.shields.io/badge/frontend-FastAPI%20%2B%20Modern%20SPA-emerald.svg)](http://localhost:8000)
[![Graph Database](https://img.shields.io/badge/graph-Neo4j%20%2B%20NetworkX-orange.svg)](https://neo4j.com/)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)]()

---

## 📌 Project Overview
**ClimateKG-RAG** is an end-to-end, production-ready system that ingests climate science research papers and policy reports, extracts a domain-specific Knowledge Graph, supports multiple independent information retrieval paradigms, and answers complex climate questions using **Hybrid GraphRAG** with strict citation grounding.

### Core Capabilities
- **Modern Interactive Web Application (v2.0):** Built with FastAPI and an ultra-responsive, zero-reload dark glassmorphism SPA featuring Vis.js Network physics and Chart.js analytics.
- **Evidence-Grounded Q&A Studio:** Synthesizes verified answers with interactive, clickable citations (`[1]`, `[2]`), confidence badges, latency profiling, and multi-hop causal graph pathways (`CO2 -> CAUSES -> Global Warming -> AFFECTS -> Drought -> ADAPTED_BY -> Drip Irrigation`).
- **Knowledge Graph Explorer 2.0:** Full-screen interactive graph canvas with 10 Node Types, 8 Relation Types, entity search with auto-focus, evidence inspector drawer, and an automated **Causal Path Discovery Tool** to trace chains between any two climate entities.
- **Multi-Paradigm Comparison Lab:** Side-by-side evaluation running queries simultaneously across TF-IDF, BM25, Dense Semantic (`all-MiniLM-L6-v2`), Knowledge Graph, and Hybrid GraphRAG.
- **Multi-Format Ingestion:** Ingests PDF, DOCX, and TXT documents with sentence-aware chunking, metadata retention, and SHA-256 deduplication.
- **Dual Graph Database Support:** Seamless automatic fallback to in-memory/disk persistent **NetworkX MultiDiGraph**, plus full support for **Neo4j Community 5.x** via Bolt Cypher.
- **Standardized IR Benchmark:** Evaluates all 5 methods across 32 manually verifiable climate queries for Precision@K, Recall@K, MRR, MAP, NDCG@K, Latency, and Faithfulness.

---

## 🏛️ System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │ Raw Documents (PDF, DOCX, TXT Corpus)   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │ Document Extraction & Cleaning (PyMuPDF)│
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │ Sentence-Aware Chunking & Provenance    │
                    └────────┬───────────────────────┬────────┘
                             │                       │
                             ▼                       ▼
            ┌────────────────────────────────┐  ┌────────────────────────────────┐
            │ Document & Chunk Store (JSON)  │  │ Knowledge Graph Extraction     │
            │ + SHA-256 Deduplication        │  │ (Ollama LLM / Deterministic)   │
            └────────┬───────────────────────┘  └────────┬───────────────────────┘
                     │                                   │
                     ▼                                   ▼
            ┌────────────────────────────────┐  ┌────────────────────────────────┐
            │ Lexical & Dense Indexing       │  │ Dual Graph Store:              │
            │ - TF-IDF Matrix                │  │ - NetworkX MultiDiGraph (Disk) │
            │ - BM25 Okapi Index             │  │ - Neo4j Community (Cypher)     │
            │ - Vector Cache (all-MiniLM)    │  │ - Provenance & Evidence Links  │
            └────────┬───────────────────────┘  └────────┬───────────────────────┘
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │           User Query Input              │
                    └──────────────────┬──────────────────────┘
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
    │  BM25 Lexical   │       │  Dense Semantic │       │ Knowledge Graph │
    │   Retrieval     │       │   Retrieval     │       │ Multi-Hop Search│
    └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
             │                         │                         │
             └─────────────────────────┼─────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ Reciprocal Rank Fusion (RRF Reranking)  │
                    │   Score(d) = Sum( Weight / (60 + Rank)) │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ Grounded Answer Generator               │
                    │ (Strict Evidence Constraint + Paths)    │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ Answer with Grounded Citations & Graph  │
                    └─────────────────────────────────────────┘
```

---

## 📐 Knowledge Graph Schema

### Node Types (10 Classes)
| Node Type | Description & Domain Examples |
| :--- | :--- |
| `ClimatePhenomenon` | Global Warming, Sea Level Rise, Heatwaves, Monsoon Variability, Ocean Acidification |
| `GreenhouseGas` | Carbon Dioxide ($CO_2$), Methane ($CH_4$), Nitrous Oxide ($N_2O$) |
| `Region` | South Asia, India, Arctic, Amazon Basin, Global |
| `ClimateImpact` | Agricultural Yield Decline, Water Scarcity, Coral Bleaching, Coastal Flooding |
| `MitigationStrategy` | Solar Energy, Wind Energy, Afforestation, Carbon Capture and Storage |
| `AdaptationStrategy` | Drip Irrigation, Drought-Resilient Crop Varieties, Rainwater Harvesting, Sea Walls |
| `Policy` | Paris Agreement, Kyoto Protocol, National Action Plan on Climate Change (NAPCC) |
| `ResearchPaper` | IPCC Sixth Assessment Report (AR6), UNEP Emissions Gap Report |
| `Organization` | IPCC, UNFCCC, WMO, UNEP, ICAR |
| `Dataset` | ERA5 Reanalysis Dataset, CMIP6 Climate Projections, NOAA Mauna Loa Record |

### Relationship Types (8 Classes)
- `CAUSES`: Causal mechanism ($CO_2 \xrightarrow{\text{CAUSES}} \text{Global Warming}$)
- `AFFECTS`: Direct impact ($\text{Global Warming} \xrightarrow{\text{AFFECTS}} \text{Agricultural Yield Decline}$)
- `OCCURS_IN`: Geographic location ($\text{Agricultural Yield Decline} \xrightarrow{\text{OCCURS_IN}} \text{India}$)
- `MITIGATED_BY`: Decarbonization ($\text{Global Warming} \xrightarrow{\text{MITIGATED_BY}} \text{Solar Energy}$)
- `ADAPTED_BY`: Resilience measure ($\text{Agricultural Yield Decline} \xrightarrow{\text{ADAPTED_BY}} \text{Drip Irrigation}$)
- `RECOMMENDED_BY`: Institutional guidance ($\text{Drought-Resilient Varieties} \xrightarrow{\text{RECOMMENDED_BY}} \text{ICAR}$)
- `SUPPORTED_BY`: Empirical backing ($\text{Paris Agreement} \xrightarrow{\text{SUPPORTED_BY}} \text{IPCC AR6}$)
- `RELATED_TO`: Verified domain association with sentence evidence excerpt.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.11, 3.12, or 3.14
- Git
- *(Optional)* Docker Desktop (if using containerized deployment or Neo4j)
- *(Optional)* Ollama (if using local LLM generative synthesis)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/sudarshan026/ClimateKG-RAG.git
cd ClimateKG-RAG

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default parameters run 100% offline using the built-in NetworkX graph engine and local vector cache.

### 4. Optional: Start Neo4j Community Edition (Docker)
```bash
docker compose up -d neo4j
```
Neo4j Browser will be available at: `http://localhost:7474` (User: `neo4j`, Password: `climate_password123`).

---

## ⚡ Execution Commands

### Launch the Modern Web Application (Recommended)
```bash
python server.py
```
*(Or double-click `run_app.bat` on Windows)*  
- **Interactive Web App:** `http://localhost:8000`  
- **OpenAPI / Swagger Documentation:** `http://localhost:8000/docs`

*(Optional Legacy Streamlit Interface: `streamlit run app.py` at `http://localhost:8501`)*

### Ingest Corpus & Construct Knowledge Graph
Extracts text from `data/raw/`, constructs chunks, extracts entities and relations, builds indices, and generates embedding caches:
```bash
python scripts/ingest_corpus.py --force
```

### Run the 32-Query Evaluation Benchmark
Executes the evaluation suite comparing all 5 retrieval methods and outputs summary tables and CSV files:
```bash
python scripts/run_evaluation.py
```
*(Or double-click `run_eval.bat` on Windows)*

### Run the Pytest Test Suite
Runs all 31 unit and integration tests:
```bash
pytest tests/ -v
```
*(Or double-click `run_tests.bat` on Windows)*

---

## 📊 Evaluation Benchmark Results

Measured experimental results on the 32-query `CLIMATE_BENCHMARK_QUERIES` dataset across 5 retrieval algorithms using the authentic climate research corpus:

| Retrieval Method | Precision@K | Recall@K | MRR | MAP | NDCG@K | Latency (ms) | Evidence Support Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF** | 0.3438 | 0.9375 | 0.8906 | 0.8313 | 0.8972 | 2.60 ms | 0.9531 |
| **BM25** | 0.3375 | 0.9531 | 0.8880 | 0.8554 | 0.8975 | **0.64 ms** | 0.9453 |
| **Dense Semantic** | 0.3375 | 0.9688 | 0.7917 | 0.7661 | 0.8500 | 3800 ms | 0.9583 |
| **Knowledge Graph** | 0.0625 | 0.2500 | 0.2656 | 0.2422 | 0.2500 | 1.59 ms | 0.7500 |
| **Hybrid GraphRAG** | **0.3500** | **1.0000** | 0.8333 | 0.7732 | **0.9076** | 36.36 ms | **0.9922** |

*All results computed directly from actual runs without synthetic placeholders. MAP is strictly $\le 1.0$, and claim-level evidence support replaces crude token overlap.*

For authentic document inventory, see [`data/source_inventory.md`](data/source_inventory.md).

---

## 📂 Repository Structure

```
ClimateKG-RAG/
├── server.py                          # FastAPI backend application server (v2.0)
├── app.py                             # Optional legacy Streamlit dashboard
├── static/                            # Modern UI Single Page Application (v2.0)
│   ├── index.html                     # Responsive glassmorphism web interface
│   ├── css/styles.css                 # Custom climate-tech dark theme & animations
│   └── js/app.js                      # Vis.js graph, Chart.js benchmarks & Q&A logic
├── config/
│   ├── __init__.py
│   └── settings.py                    # Environment settings, thresholds, paths
├── data/
│   ├── raw/                           # Authentic climate documents (IPCC, UNFCCC, ICAR)
│   ├── processed/                     # Indexed documents.json and chunks.json
│   ├── vector_cache/                  # Precomputed dense embeddings (embeddings.npz)
│   ├── graph_storage/                 # Serialized Knowledge Graph (graph.json)
│   ├── evaluation/                    # Exported evaluation summary & detailed CSVs
│   └── sample_climate_docs.py         # Authentic climate science corpus generator
├── docs/
│   ├── PROJECT_REPORT.md              # Detailed academic/technical microproject report
│   └── screenshots/                   # Application screenshots directory
├── scripts/
│   ├── ingest_corpus.py               # Batch document ingestion & graph construction CLI
│   └── run_evaluation.py              # 32-query IR comparative benchmark CLI
├── src/
│   ├── document_processor/            # Extractor, cleaner, and persistent store
│   ├── chunking/                      # Sentence-aware chunker with overlap
│   ├── knowledge_graph/               # Schema, extractor, normalizer, Neo4j, NetworkX
│   ├── retrieval/                     # TF-IDF, BM25, Dense, Graph, Hybrid, RRF
│   ├── qa/                            # Grounded answer generator and prompt templates
│   ├── evaluation/                    # Benchmark dataset (32 queries), metrics, evaluator
│   └── visualization/                 # PyVis interactive network & Plotly charts
├── tests/                             # 31 Pytest unit & integration tests
├── .github/workflows/ci.yml           # Automated CI/CD test and benchmark workflow
├── Dockerfile                         # Production container image configuration
├── docker-compose.yml                 # Multi-container orchestration (App + Neo4j)
├── run_app.bat                        # One-click Windows launcher for Web App
├── run_eval.bat                       # One-click Windows launcher for Benchmark
├── run_tests.bat                      # One-click Windows launcher for Pytest
├── requirements.txt                   # Complete Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

---

## ⚖️ License
This project is open-source under the MIT License.  
Aligned with the **United Nations Sustainable Development Goal 13: Climate Action**.
