"""
Centralized application configuration.

All tunables are environment-driven (12-factor style) so the same image
can be promoted from dev -> staging -> prod without code changes.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---
    app_name: str = "academic-lit-review-agent"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- LLM provider ---
    llm_provider: Literal["anthropic", "ollama", "groq"] = "anthropic"
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: str = Field(default="", description="Required for real runs")
    groq_api_key: str = Field(default="", description="Required for Groq runs")
    llm_model: str = "claude-sonnet-4-6"
    local_llm_model: str = "phi3:latest"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.2
    llm_max_retries: int = 4
    llm_timeout_seconds: float = 90.0

    # --- Decomposition / fan-out ---
    max_sub_questions: int = 5
    max_papers_per_sub_question: int = 8
    max_concurrent_research_agents: int = 5
    max_feedback_loop_iterations: int = 2
    max_decomposition_clusters: int = 5

    # --- ArXiv ---
    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_min_request_interval_seconds: float = 3.1  # ArXiv asks for >=3s between requests
    arxiv_max_results_per_query: int = 15

    # --- Web search (pluggable provider) ---
    web_search_provider: str = "duckduckgo"
    brave_api_key: str = ""
    serpapi_api_key: str = ""
    scholarapi_api_key: str = ""
    web_search_max_results: int = 8

    # --- PDF fetching ---
    pdf_fetch_timeout_seconds: float = 30.0
    pdf_max_bytes: int = 25_000_000  # 25MB safety cap
    pdf_max_chars_for_llm: int = 60_000  # truncate extracted text before sending to LLM

    # --- HTTP client ---
    http_max_retries: int = 3
    http_backoff_base_seconds: float = 1.5

    # --- Storage ---
    sqlite_path: str = "./data/lit_review.db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: str = "./data/embeddings_cache"
    dedup_similarity_threshold: float = 0.93

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_body_max_bytes: int = 50_000
    rate_limit_requests_per_minute: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
