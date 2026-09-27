"""
Configuración centralizada de la aplicación.
Todas las variables sensibles/ambientales se leen desde .env
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "RAG Multimodal Service"
    ENV: str = "development"

    # --- LLM Provider: "openai" | "ollama" | "gemini" ---
    LLM_PROVIDER: str = "ollama"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # --- Embeddings ---
    EMBEDDING_PROVIDER: str = "ollama"  # "openai" | "ollama" | "gemini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # --- Vector store ---
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "documents"

    # --- Storage ---
    UPLOAD_DIR: str = "./data/uploads"
    IMAGES_DIR: str = "./data/images"

    # --- Retrieval ---
    TOP_K: int = 5
    CHUNK_MAX_TOKENS: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Cacheado: se lee el .env una sola vez por proceso."""
    return Settings()
