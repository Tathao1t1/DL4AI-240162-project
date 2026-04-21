"""
fastapi-users router setup — JWT auth (login/register/me).

Secret is read from env var JWT_SECRET (default: dev-secret-change-in-prod).
"""
import os
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import BeanieUserDatabase

from api.auth.models import User
from api.auth.schemas import UserCreate, UserRead, UserUpdate

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-prod")
JWT_LIFETIME = int(os.getenv("JWT_LIFETIME_SECONDS", 60 * 60 * 24 * 7))  # 7 days


async def get_user_db():
    yield BeanieUserDatabase(User)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=JWT_LIFETIME)


bearer_transport = BearerTransport(tokenUrl="/api/v1/auth/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, str](
    get_user_db,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)

# Router bundle: register + login + me
auth_router = fastapi_users.get_auth_router(auth_backend)
register_router = fastapi_users.get_register_router(UserRead, UserCreate)
users_router = fastapi_users.get_users_router(UserRead, UserUpdate)
