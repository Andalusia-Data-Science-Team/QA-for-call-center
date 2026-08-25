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
    llm_model: str = "z-ai/glm-5.3"

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    huggingface_api_key: str = ""
    # Match the sibling *_api_key fields above: default to "" and let
    # pydantic_settings populate it from the env var itself. The previous
    # `os.getenv("OPENROUTER_API_KEY")` default evaluated to None whenever
    # that var was unset, which fails validation against the `str` type and
    # crashes Settings() (and therefore the whole app) at import time.
    openrouter_api_key: str = ""

    # Retry config 
    llm_max_retries: int = 5
    llm_retry_delay_seconds: float = 1.5   # base delay; doubles each attempt

    # ── Performance ───────────────────────────────────────────────────────────
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 60.0

     
    # --- Dynamics 365 CRM (doctor reference data incl. walk-in / cash price) ---
    # Accessed via the CRM TDS endpoint (SQL protocol) with Azure AD auth.
    CRM_SERVER: str = os.getenv("CRM_SERVER", "")                 # e.g. "org2f45e702.crm4.dynamics.com,5558"
    CRM_CLIENT_ID: str = os.getenv("CRM_CLIENT_ID", "51f81489-12ee-4a9e-aaae-a2591f45987d")
    CRM_TENANT: str = os.getenv("CRM_TENANT", "organizations")
    CRM_USERNAME: str = os.getenv("CRM_USERNAME", "")
    CRM_PASSWORD: str = os.getenv("CRM_PASSWORD", "")
    CRM_DOCTOR_TABLE: str = os.getenv("CRM_DOCTOR_TABLE", "dbo.cr301_newdoctordataset")
    CRM_OFFER_TABLE: str = os.getenv("CRM_OFFER_TABLE", "new_offer_equest")
    CRM_FEE_TABLE: str = os.getenv("CRM_FEE_TABLE", "dbo.cr301_table1")
    CRM_PRICE_CACHE_TTL_SECONDS: int = int(os.getenv("CRM_PRICE_CACHE_TTL_SECONDS", "86400"))  # 24h
    DB_DRIVER: str = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")


settings = Settings()
