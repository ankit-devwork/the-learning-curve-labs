from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import UnauthorizedException


class AuthUser(BaseModel):
    id: str
    email: str | None = None
    role: str | None = None


def decode_supabase_jwt(token: str) -> AuthUser:
    if not settings.supabase_jwt_secret.strip():
        raise UnauthorizedException("SUPABASE_JWT_SECRET is not configured on the backend")

    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except JWTError as exc:
        raise UnauthorizedException("Invalid or expired token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Token missing subject (sub)")

    return AuthUser(
        id=str(user_id),
        email=payload.get("email"),
        role=payload.get("role"),
    )
