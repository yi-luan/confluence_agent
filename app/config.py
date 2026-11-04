"""集中管理設定：全部從環境變數 / .env 讀取，方便在 Docker、CI、雲端之間切換。"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "azure", "gemini", "ollama", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"
    admin_api_key: str = "change-me"  # 保護 /ingest 等管理端點

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/kb"

    # --- LLM / Embedding provider（可切換） ---
    llm_provider: Provider = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 60.0

    embedding_provider: Provider = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024  # 換 embedding 模型時要一起改，並重建索引

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_chat_deployment: str | None = None
    azure_embedding_deployment: str | None = None

    google_api_key: str | None = None  # Gemini（Google AI Studio）

    ollama_base_url: str = "http://localhost:11434"

    # --- RAG 參數 ---
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.35  # cosine similarity 低於此值視為不相關
    retrieval_min_keyword: float = 0.3  # 或問題關鍵詞有 30% 以上出現在片段中
    chunk_size: int = 400
    chunk_overlap: int = 80
    history_turns: int = 6

    # --- Confluence ---
    confluence_url: str | None = None
    confluence_user: str | None = None  # Cloud 版填 email；Server/DC 用 PAT 時留空
    confluence_token: str | None = None
    confluence_spaces: str = ""  # 逗號分隔，例如 "ENG,HR"

    @property
    def confluence_space_list(self) -> list[str]:
        return [s.strip() for s in self.confluence_spaces.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
