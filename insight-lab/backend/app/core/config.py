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

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "insightlab_dev_password"

    redis_host: str = ""
    redis_port: int = 6379
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    groq_api_key: str = ""


settings = Settings()
