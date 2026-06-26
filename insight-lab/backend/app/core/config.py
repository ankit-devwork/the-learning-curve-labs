import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BACKEND_DIR / ".env"
ROOT_ENV_PATH = BACKEND_DIR.parent / ".env"


def _apply_env_file(path: Path) -> None:
    """Load variables from a .env file when OS env is unset or blank."""
    if not path.is_file():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return

    for key, value in dotenv_values(path, encoding="utf-8-sig").items():
        if value is None or not str(value).strip():
            continue
        current = os.environ.get(key)
        if current is None or not str(current).strip():
            os.environ[key] = value


# Fill gaps from backend/.env, then repo-root .env (common on Windows setups).
_apply_env_file(ENV_PATH)
_apply_env_file(ROOT_ENV_PATH)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8-sig",
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

    document_storage_encryption_key: str = ""

    retry_max_attempts: int = 4
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_sec: float = 60.0


settings = Settings()


def config_diagnostics() -> dict[str, object]:
    """Non-secret startup hints for /ready and logs."""
    url = settings.supabase_url.strip()
    return {
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.is_file(),
        "root_env_file": str(ROOT_ENV_PATH),
        "root_env_file_exists": ROOT_ENV_PATH.is_file(),
        "supabase_url_configured": bool(url),
        "supabase_url_host": url.split("//")[1].split("/")[0] if "//" in url else None,
        "supabase_service_role_configured": bool(settings.supabase_service_role_key.strip()),
    }
