from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_debug: bool = True
    app_name: str = "InsightLab API"
    log_dir: str = "logs"

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "insightlab_dev_password"

    redis_host: str = ""
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    groq_api_key: str = ""

    storage_bucket: str = "uploads"
    upload_max_bytes: int = 20 * 1024 * 1024  # 20 MB
    rate_limit_upload_per_hour: int = 10


settings = Settings()
