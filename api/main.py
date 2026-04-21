"""
QuantPulse AI — FastAPI backend
Serves Task 2 predictions, Task 3 signals, Task 4 portfolio data,
OHLCV market data, and an automated daily pipeline.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, signals, portfolio, market, pipeline, predict_nasdaq, search, live
from api.auth.router import router as auth_router
from api.scheduler import start_scheduler
from api import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting QuantPulse API")
    start_scheduler()
    yield
    # Shutdown
    await database.close()
    logger.info("QuantPulse API shut down")


app = FastAPI(
    title="QuantPulse AI",
    description="Vietnam stock market prediction API — Tasks 2, 3, 4",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
app.include_router(auth_router,            prefix=PREFIX)


@app.get("/")
async def root():
    return {
        "service": "QuantPulse AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
