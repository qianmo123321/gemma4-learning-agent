from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    app_name = os.getenv("APP_NAME", "Gemma4 Private Learning Agent API")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    cors_origins = [x.strip() for x in os.getenv(
        "CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    ).split(",") if x.strip()]

    model_provider = os.getenv("MODEL_PROVIDER", "openai_compatible").lower()
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    vllm_model = os.getenv("VLLM_MODEL", "google/gemma-4-12b-it")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

    rag_top_k_default = int(os.getenv("RAG_TOP_K_DEFAULT", "3"))
    rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "90"))
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "30"))

    knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", str(BACKEND_DIR / "data" / "knowledge")))
    rag_index_path = Path(os.getenv("RAG_INDEX_PATH", str(BACKEND_DIR / "data" / "hybrid_rag.joblib")))
    chat_log_path = Path(os.getenv("CHAT_LOG_PATH", str(BACKEND_DIR / "data" / "qa_history.jsonl")))
    feedback_log_path = Path(os.getenv("FEEDBACK_LOG_PATH", str(BACKEND_DIR / "data" / "feedback.jsonl")))
    conversation_db_path = Path(
        os.getenv(
            "CONVERSATION_DB_PATH",
            str(BACKEND_DIR / "data" / "learning_agent.db"),
        )
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chat_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.conversation_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
