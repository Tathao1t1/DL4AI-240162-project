"""
/api/v1/search — ticker fuzzy search over a static registry.

GET /search?q=fp  → returns up to 10 tickers matching symbol or name.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter(prefix="/search", tags=["search"])

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "ticker_registry.json"

# Load once at import time
with open(_REGISTRY_PATH) as _f:
    _REGISTRY: list[dict] = json.load(_f)


@router.get("")
async def search_tickers(q: str = Query(..., min_length=1)):
    """
    Case-insensitive substring search across symbol and name.
    Returns up to 10 matching entries.
    """
    q_lower = q.strip().lower()
    results = [
        entry for entry in _REGISTRY
        if q_lower in entry["symbol"].lower() or q_lower in entry["name"].lower()
    ]
    return results[:10]
