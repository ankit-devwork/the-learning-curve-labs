from functools import lru_cache

import jwt as pyjwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError
from jose import JWTError, jwt as jose_jwt
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import UnauthorizedException


class AuthUser(BaseModel):
    id: str
    email: str | None = None
    role: str | None = None


@lru_cache(maxsize=1)
def _get_jwks_client(supabase_url: str) -> PyJWKClient:
    jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url, cache_keys=True)


def _payload_from_token(token: str) -> dict:
    try:
        header = pyjwt.get_unverified_header(token)
    except Exception as exc:
        raise UnauthorizedException("Malformed token") from exc

    algorithm = header.get("alg", "HS256")

    # Legacy Supabase projects — symmetric HS256 with JWT secret
    if algorithm == "HS256":
        if not settings.supabase_jwt_secret.strip():
            raise UnauthorizedException(
                "SUPABASE_JWT_SECRET is not configured (required for HS256 tokens)"
            )
        try:
            return jose_jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except JWTError as exc:
            raise UnauthorizedException(_jwt_error_message(exc)) from exc

    # New Supabase signing keys — asymmetric ES256/RS256 via JWKS
    if not settings.supabase_url.strip():
        raise UnauthorizedException(
            "SUPABASE_URL is not configured (required for JWKS token verification)"
        )

    issuer = f"{settings.supabase_url.rstrip('/')}/auth/v1"
    try:
        signing_key = _get_jwks_client(settings.supabase_url).get_signing_key_from_jwt(token).key
        return pyjwt.decode(
            token,
            signing_key,
            algorithms=[algorithm],
            audience="authenticated",
            issuer=issuer,
        )
    except InvalidTokenError as exc:
        raise UnauthorizedException(_jwt_error_message(exc)) from exc


def _jwt_error_message(exc: Exception) -> str:
    _ = exc
    return "Invalid or expired token"


def decode_supabase_jwt(token: str) -> AuthUser:
    payload = _payload_from_token(token)

    role = payload.get("role")
    if role != "authenticated":
        raise UnauthorizedException("Invalid token role")

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Token missing subject (sub)")

    return AuthUser(
        id=str(user_id),
        email=payload.get("email"),
        role=payload.get("role"),
    )
