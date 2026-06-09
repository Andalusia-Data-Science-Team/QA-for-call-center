from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider
    # Accepted values: "anthropic" | "openai" | "huggingface"
    # Used huggingface for demo because of free tokens
    llm_provider: str = "openrouter"

    # Anthropic default: claude-sonnet-4-20250514
    # OpenAI default:    gpt-4o
    # HuggingFace default: Qwen/Qwen2.5-7B-Instruct
    llm_model: str = "openai/gpt-oss-120b:free"

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    huggingface_api_key: str = ""
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")

    # Retry config 
    llm_max_retries: int = 5
    llm_retry_delay_seconds: float = 1.5   # base delay; doubles each attempt

    # ── Performance ───────────────────────────────────────────────────────────
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 60.0


settings = Settings()
