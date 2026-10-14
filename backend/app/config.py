import pathlib
from typing import Literal

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime ---
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    # Interactive API docs (/docs, /redoc, /openapi.json). Always off in production
    # unless explicitly enabled.
    enable_api_docs: bool | None = None

    # --- Supabase / Postgres ---
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str

    # --- LLM gateway (any OpenAI-compatible endpoint; OmniRoute recommended) ---
    llm_api_key: str
    llm_base_url: str | None = None
    llm_chat_model: str = "gpt-4.1"
    llm_grounding_model: str = "gpt-4.1-mini"
    # Short, cheap calls (thread titles, keyword extraction). Falls back to
    # llm_grounding_model when unset.
    llm_fast_model: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # How the main agent returns its typed answer:
    #   tool     - function-calling output (best with OpenAI / strong tool-capable models)
    #   prompted - JSON requested in the prompt and parsed (works with weaker/free models)
    llm_output_mode: Literal["tool", "prompted"] = "tool"
    # Structured output strategy for single-shot helper calls (grounding judge,
    # keyword extraction). The gateway auto-degrades json_schema -> json_object ->
    # prompt-only when a model or provider rejects the stricter mode.
    llm_structured_mode: Literal["json_schema", "json_object", "prompt"] = "json_schema"

    openai_agent_request_limit: int = 20
    openai_agent_temperature: float = 0.0

    # --- Embeddings ---
    # Defaults to the OpenAI API. Point these at any OpenAI-compatible /v1/embeddings
    # endpoint (OmniRoute included). NOTE: the pgvector column is vector(1536);
    # changing the model/dimension requires re-embedding the whole corpus.
    openai_api_key: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_embedding_timeout_seconds: float = 10.0
    embedding_max_retries: int = 2
    # Some providers reject the `dimensions` parameter. Disable to omit it.
    embedding_send_dimensions: bool = True

    # --- Retrieval ---
    retrieval_candidate_k: int = 50
    retrieval_top_k: int = 10
    retrieval_rrf_k: int = 60
    retrieval_neighbor_radius: int = 1
    retrieval_fts_config: str = "english"
    retrieval_fts_keyword_model: str = "gpt-4.1-mini"
    retrieval_fts_keyword_min: int = 3
    retrieval_fts_keyword_max: int = 5
    retrieval_fts_keyword_fast_path_tokens: int = 5

    # --- Abuse protection ---
    rate_limit_enabled: bool = True
    # Chat turns per user per rolling minute (per backend process).
    rate_limit_chat_per_minute: int = 20
    # Hard cap on characters in one user message.
    max_message_chars: int = 4000

    # Comma-separated in .env; use `cors_origins` for the parsed list.
    allowed_origins: str = "http://localhost:5173"

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if not (self.embedding_api_key or self.openai_api_key):
            raise ValueError(
                "Set EMBEDDING_API_KEY (or OPENAI_API_KEY) so query and ingestion "
                "embeddings can be generated."
            )
        if self.app_env == "production" and "*" in self.cors_origins:
            raise ValueError(
                "ALLOWED_ORIGINS must list explicit origins in production "
                "(wildcard is not allowed with credentialed requests)."
            )
        return self

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        """Normalize Supabase-style URLs for SQLAlchemy + psycopg v3."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def api_docs_enabled(self) -> bool:
        if self.enable_api_docs is not None:
            return self.enable_api_docs
        return not self.is_production

    @property
    def resolved_embedding_api_key(self) -> str:
        key = self.embedding_api_key or self.openai_api_key
        assert key  # guaranteed by _validate
        return key

    @property
    def resolved_fast_model(self) -> str:
        return self.llm_fast_model or self.llm_grounding_model


settings = Settings()
