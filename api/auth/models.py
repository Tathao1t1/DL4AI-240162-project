"""
Beanie User document — extends fastapi-users BeanieBaseUser.
Adds watchlist and alert_prefs for future Phase 2 features.
"""
from typing import List
from beanie import Document
from fastapi_users.db import BeanieBaseUser
from pydantic import Field


class User(BeanieBaseUser, Document):
    watchlist:   List[str] = Field(default_factory=list)
    alert_prefs: dict      = Field(default_factory=dict)

    class Settings:
        name = "users"
