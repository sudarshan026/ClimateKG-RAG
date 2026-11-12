import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Force offline mode for Hugging Face Hub to use local cache and prevent network timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
    # Project paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    VECTOR_CACHE_DIR: Path = BASE_DIR / "data" / "vector_cache"
    EVALUATION_DIR: Path = BASE_DIR / "data" / "evaluation"
    GRAPH_STORAGE_DIR: Path = BASE_DIR / "data" / "graph_storage"
    
    # Document Processing & Chunking
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "250"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
    SUPPORTED_EXTENSIONS: list[str] = field(default_factory=lambda: [".pdf", ".txt", ".docx"])
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "climate_password123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    USE_NEO4J_IF_AVAILABLE: bool = os.getenv("USE_NEO4J_IF_AVAILABLE", "true").lower() in ("true", "1", "yes")
    
    # Ollama / LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    USE_OLLAMA_IF_AVAILABLE: bool = os.getenv("USE_OLLAMA_IF_AVAILABLE", "true").lower() in ("true", "1", "yes")
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    GRAPH_MAX_HOPS: int = int(os.getenv("GRAPH_MAX_HOPS", "2"))
    
    def ensure_directories(self):
        """Create necessary directories if they do not exist."""
        for d in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.VECTOR_CACHE_DIR,
            self.EVALUATION_DIR,
            self.GRAPH_STORAGE_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
