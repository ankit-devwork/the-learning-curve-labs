from functools import lru_cache

from supabase import Client, create_client

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    if not settings.supabase_url.strip() or not settings.supabase_service_role_key.strip():
        raise ServiceUnavailableException("Supabase is not configured on the backend")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
