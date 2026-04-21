"""
Pydantic schemas for fastapi-users endpoints (register, read, update).
"""
from typing import Optional
from fastapi_users import schemas


class UserRead(schemas.BaseUser[str]):
    pass


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass
