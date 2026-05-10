"""
TensorFinance — FastAPI backend
Serves Task 2 predictions, Task 3 signals, Task 4 portfolio data,
OHLCV market data, and an automated daily pipeline.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, signals, portfolio, market, pipeline, predict_nasdaq, search, live
from api.scheduler import start_scheduler
from api import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")


async def _preload_models():
    from api.routers.predict_nasdaq import preload_all as preload_nasdaq
    from inference import preload_all as preload_vn
    await asyncio.gather(preload_nasdaq(), preload_vn(), return_exceptions=True)
    logger.info("Model preload complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting TensorFinance API")
    await database.ensure_indexes()
    start_scheduler()
    asyncio.create_task(_preload_models())
    yield
    # Shutdown
    await database.close()
    logger.info("TensorFinance API shut down")


app = FastAPI(
    title="TensorFinance",
    description="Vietnam stock market prediction API — Tasks 2, 3, 4",
    version="1.0.0",
    lifespan=lifespan,
)

_CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"

app.include_router(predict.router,         prefix=PREFIX)
app.include_router(predict_nasdaq.router,  prefix=PREFIX)
app.include_router(signals.router,         prefix=PREFIX)
app.include_router(portfolio.router,       prefix=PREFIX)
app.include_router(market.router,          prefix=PREFIX)
app.include_router(pipeline.router,        prefix=PREFIX)
app.include_router(search.router,          prefix=PREFIX)
app.include_router(live.router,            prefix=PREFIX)


@app.get("/")
async def root():
    return {
        "service": "TensorFinance",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
