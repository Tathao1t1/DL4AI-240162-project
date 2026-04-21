"""
Motor (async MongoDB) connection — shared across all routers.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "quantpulse")

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


async def close():
    global _client
    if _client:
        _client.close()
        _client = None
