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
    CRM_LEADS_LLM_MAX_TOKENS: int = max(2048, int(os.getenv("CRM_LEADS_LLM_MAX_TOKENS", "4096")))
    OVERALL_SCORING_LLM_MAX_TOKENS: int = max(2048, int(os.getenv("OVERALL_SCORING_LLM_MAX_TOKENS", "4096")))
    LLM_JSON_PARSE_RETRIES: int = max(0, int(os.getenv("LLM_JSON_PARSE_RETRIES", "1")))
    LLM_INPUT_COST_PER_MILLION_USD: float = max(0.0, float(os.getenv("LLM_INPUT_COST_PER_MILLION_USD", "0")))
    LLM_OUTPUT_COST_PER_MILLION_USD: float = max(0.0, float(os.getenv("LLM_OUTPUT_COST_PER_MILLION_USD", "0")))
    # Dedicated JSONL telemetry files. TOKEN_USAGE_LOG_PATH remains for
    # backwards compatibility with callers of the former combined logger.
    TOKEN_USAGE_LOG_PATH: str = os.getenv("TOKEN_USAGE_LOG_PATH", "logs/token_usage.log")
    NODE_CONSUMPTION_LOG_PATH: str = os.getenv(
        "NODE_CONSUMPTION_LOG_PATH", "logs/node_consumption.log"
    )
    OVERALL_CONSUMPTION_LOG_PATH: str = os.getenv(
        "OVERALL_CONSUMPTION_LOG_PATH", "logs/overall_consumption.log"
    )
    NODE_OUTPUT_LOG_PATH: str = os.getenv("NODE_OUTPUT_LOG_PATH", "logs/node_output.log")
    TOKEN_USAGE_LOG_MAX_BYTES: int = max(1024, int(os.getenv("TOKEN_USAGE_LOG_MAX_BYTES", "10485760")))
    TOKEN_USAGE_LOG_BACKUP_COUNT: int = max(1, int(os.getenv("TOKEN_USAGE_LOG_BACKUP_COUNT", "5")))
    llm_timeout_seconds: float = 60.0

     
    # --- Dynamics 365 CRM (doctor reference data incl. walk-in / cash price) ---
    # Doctor and CRM Leads data use the Dataverse TDS endpoint; CRM Leads uses
    # OAuth client credentials supplied through its dedicated environment keys.
    CRM_SERVER: str = os.getenv("CRM_SERVER", "")                 # e.g. "org2f45e702.crm4.dynamics.com,5558"
    CRM_CLIENT_ID: str = os.getenv("CRM_CLIENT_ID", "51f81489-12ee-4a9e-aaae-a2591f45987d")
    CRM_TENANT: str = os.getenv("CRM_TENANT", "organizations")
    CRM_USERNAME: str = os.getenv("CRM_USERNAME", "")
    CRM_PASSWORD: str = os.getenv("CRM_PASSWORD", "")
    CRM_DOCTOR_TABLE: str = os.getenv("CRM_DOCTOR_TABLE", "dbo.cr301_newdoctordataset")
    CRM_OFFER_TABLE: str = os.getenv("CRM_OFFER_TABLE", "new_offer_equest")
    CRM_FEE_TABLE: str = os.getenv("CRM_FEE_TABLE", "dbo.cr301_table1")
    CRM_LEADS_SERVER: str = os.getenv("CRM_LEADS_SERVER", "")
    CRM_LEADS_TABLE: str = os.getenv("CRM_LEADS_TABLE", "dbo.lead")
    CRM_LEADS_MAX_RETRIES: int = max(1, int(os.getenv("CRM_LEADS_MAX_RETRIES", "4")))
    CRM_LEADS_RETRY_DELAY_SECONDS: float = max(0.0, float(os.getenv("CRM_LEADS_RETRY_DELAY_SECONDS", "1")))
    CRM_APP_NAME: str = os.getenv("CRM_App_Name", os.getenv("CRM_APP_NAME", ""))
    CRM_TENANT_ID: str = os.getenv("CRM_Tenant_Id", os.getenv("CRM_TENANT_ID", ""))
    CRM_LEAD_CLIENT_ID: str = os.getenv("CRM_LEAD_CLIENT_ID", "")
    CRM_SECRET_ID: str = os.getenv("CRM_Secret_ID", os.getenv("CRM_SECRET_ID", ""))
    CLIENT_SECRET: str = os.getenv("CLIENT_SECRET", "")
    CRM_PRICE_CACHE_TTL_SECONDS: int = int(os.getenv("CRM_PRICE_CACHE_TTL_SECONDS", "86400"))  # 24h
    DB_DRIVER: str = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")


settings = Settings()
