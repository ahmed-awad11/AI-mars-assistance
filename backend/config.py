"""
Marswalk AI Assistant - Configuration
Centralized configuration management using Pydantic Settings (v2).
"""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Marswalk AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "qwen3:1.7b"
    MODEL_TIMEOUT: int = 120  # seconds

    # Embedding Model (optional - for RAG)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # Database
    DATABASE_URL: str = "sqlite:///./data/marswalk_ai.db"

    # ChromaDB (optional - for RAG)
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "marswalk_knowledge"

    # Document Ingestion
    DOCUMENTS_DIR: str = "./data/documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Conversation
    MAX_HISTORY_MESSAGES: int = 20
    MAX_CONTEXT_TOKENS: int = 4000

    # Tool Calling
    ENABLE_TOOLS: bool = True
    TOOL_TIMEOUT: int = 30

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*"  # Allow all for development
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()

# Ensure data directories exist
Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)