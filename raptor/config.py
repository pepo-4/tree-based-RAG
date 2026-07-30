import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Config:
    """Global configuration settings for RAPTOR pipeline."""
    
    # Gemini API Keys for Rotation
    @staticmethod
    def get_gemini_api_keys() -> List[str]:
        keys = []
        env_keys = os.getenv("GEMINI_API_KEYS")
        if env_keys:
            keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
            
        idx = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{idx}")
            if not key:
                break
            keys.append(key.strip())
            idx += 1
            
        single_key = os.getenv("GEMINI_API_KEY")
        if single_key and single_key not in keys:
            keys.append(single_key.strip())
            
        return keys

    # Ollama Host & Model Settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

    # Embedding & Reranking Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # Vector Storage Path
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index_all_texts")

    # Neo4j Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Pipeline Hyperparameters (Matching RAPTOR notebook & paper specifications)
    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 50
    CUMULATIVE_PROBABILITY_THRESHOLD: float = 0.90
    PCA_VARIANCE_RATIO: float = 0.90      # Retain 90% variance in PCA
    GLOBAL_UMAP_DIM: int = 10
    LOCAL_UMAP_DIM: int = 10
    LOCAL_UMAP_NEIGHBORS: int = 10        # Fixed local neighbors=10 as in RAPTOR
