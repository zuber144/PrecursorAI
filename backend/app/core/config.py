from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/precursorai"

    # Gemini
    GEMINI_API_KEY: str = ""

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Tier 1
    CONFIDENCE_THRESHOLD: float = 0.75
    RAG_TOP_K: int = 5

    # Tier 2
    COGNITION_SIMILARITY_THRESHOLD: float = 0.80
    COGNITION_MIN_REPORTS: int = 3


settings = Settings()
