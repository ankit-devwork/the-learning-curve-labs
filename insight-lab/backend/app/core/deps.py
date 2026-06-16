from typing import Annotated

from fastapi import Depends, Header

from app.core.auth import AuthUser, decode_supabase_jwt
from app.core.exceptions import UnauthorizedException


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise UnauthorizedException("Missing Authorization: Bearer <token> header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise UnauthorizedException("Empty bearer token")

    return decode_supabase_jwt(token)
