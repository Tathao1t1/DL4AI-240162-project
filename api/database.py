"""
Motor (async MongoDB) connection — shared across all routers.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "tensorfinance")

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client


def get_db():
    return get_client()[DB_NAME]


# Collection helpers
def predictions_col():
    return get_db()["predictions"]


def pipeline_runs_col():
    return get_db()["pipeline_runs"]


async def ensure_indexes():
    """Create indexes that don't already exist (idempotent — safe to call on every startup)."""
    db = get_db()
    await db["predictions"].create_index(
        [("ticker", 1), ("task", 1), ("run_at", -1)],
        name="ticker_task_run_at",
    )
    await db["pipeline_runs"].create_index(
        [("started_at", -1)],
        name="started_at_desc",
    )
    await db["users"].create_index(
        [("email", 1)],
        unique=True,
        name="email_unique",
    )


async def close():
    global _client
    if _client:
        _client.close()
        _client = None
