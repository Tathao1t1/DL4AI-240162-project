"""
Custom JWT auth router — Motor + python-jose + passlib.

Replaces fastapi-users (which deadlocks with beanie 2.x + motor 3.7+).

Endpoints:
  POST /api/v1/auth/register  — create account
  POST /api/v1/auth/login     — get JWT token
  GET  /api/v1/auth/me        — return current user (requires Bearer token)
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt as _bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

from api.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET   = os.getenv("JWT_SECRET",           "dev-secret-change-in-prod")
JWT_ALGO     = "HS256"
JWT_EXPIRE_H = int(os.getenv("JWT_EXPIRE_HOURS", "168"))   # 7 days

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Schemas ───────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    email: str
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()

def _verify(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode(), hashed.encode())

def _create_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_H)
    return jwt.encode({"sub": email, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGO)

async def _get_user_by_email(email: str) -> Optional[dict]:
    return await get_db()["users"].find_one({"email": email})

async def get_current_user(token: str = Depends(oauth2)) -> dict:
    """Dependency — decode token and return the user document."""
    creds_err = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        email: str = payload.get("sub")
        if not email:
            raise creds_err
    except JWTError:
        raise creds_err
    user = await _get_user_by_email(email)
    if not user:
        raise creds_err
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/register", response_model=UserOut, status_code=201)
async def register(body: UserRegister):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    existing = await _get_user_by_email(body.email)
    if existing:
        raise HTTPException(400, "An account with this email already exists.")
    doc = {
        "email":    body.email,
        "hashed_password": _hash(body.password),
        "is_active": True,
        "watchlist": [],
        "created_at": datetime.now(timezone.utc),
    }
    result = await get_db()["users"].insert_one(doc)
    return UserOut(id=str(result.inserted_id), email=body.email)


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await _get_user_by_email(form.username)   # username = email field
    if not user or not _verify(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=_create_token(user["email"]))


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return UserOut(id=str(user["_id"]), email=user["email"])
