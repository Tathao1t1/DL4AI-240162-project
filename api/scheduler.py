"""
APScheduler pipeline job — runs daily at 06:00 ICT.

Workflow:
  1a. INGEST VN   — fetch latest OHLCV via yfinance (.VN suffix), append to CSV
  1b. INGEST NASDAQ — fetch latest OHLCV via yfinance (plain symbol), append to CSV
  2. TRANSFORM — _add_vn_features() runs inside inference.py automatically
  3. PREDICT — predict_all_tickers() for all three horizons
  4. STORE   — upsert results into MongoDB predictions collection
"""
import sys, logging
from pathlib import Path
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference import predict_all_tickers, TICKERS
from api.database import predictions_col, pipeline_runs_col

logger = logging.getLogger("scheduler")

NASDAQ_DATA_DIR = ROOT / "nasdaq-historical-data"

# All NASDAQ tickers that have Task 1 models
try:
    _TASK1_DIR = ROOT / "models" / "task1_1" / "next_day" / "per_ticker"
    NASDAQ_TICKERS = sorted([p.name for p in _TASK1_DIR.iterdir() if p.is_dir()])
except Exception:
    NASDAQ_TICKERS = []

scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")


async def _ingest_ticker(ticker: str) -> bool:
    """Fetch latest daily bars via yfinance (.VN suffix) and append to CSV."""
    try:
        import yfinance as yf
        import pandas as pd

        csv_path = ROOT / "clean-historical-data-2026" / f"{ticker}_Historical.csv"
        if not csv_path.exists():
            return False

        # Row 0 = column names, Row 1 = ticker labels — skip row 1
        existing = pd.read_csv(csv_path, header=0, skiprows=[1])
        existing.columns = existing.columns.str.strip()

        # Last date already in the CSV
        last_date  = pd.to_datetime(existing["Date"].iloc[-1])
        fetch_from = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_to   = datetime.now().strftime("%Y-%m-%d")

        if fetch_from > fetch_to:
            logger.info("ingest %s: already up to date (%s)", ticker, last_date.date())
            return True

        # Fetch last 30 days via period (avoids yfinance exclusive-end-date issues)
        # then filter to only rows strictly after the last CSV date
        yf_symbol = f"{ticker}.VN"
        raw = yf.Ticker(yf_symbol).history(period="1mo", interval="1d")
        if raw.empty:
            logger.info("ingest %s: no data returned by yfinance", ticker)
            return True

        raw = raw.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        # Keep only rows newer than the last CSV entry
        new_rows = raw[raw["Date"] > last_date].copy()
        if new_rows.empty:
            logger.info("ingest %s: already up to date (%s)", ticker, last_date.date())
            return True

        new_rows["Date"] = new_rows["Date"].dt.strftime("%Y-%m-%d")
        latest = new_rows[["Date", "Close", "High", "Low", "Open", "Volume"]].copy()

        existing["Date"] = pd.to_datetime(existing["Date"]).dt.strftime("%Y-%m-%d")

        # Merge and deduplicate on Date, keep latest values
        combined = (
            pd.concat([existing, latest])
            .drop_duplicates(subset="Date", keep="last")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        combined.to_csv(csv_path, index=False)
        logger.info("ingest %s: +%d new rows (up to %s)",
                    ticker, len(latest), latest["Date"].iloc[-1])
        return True
    except Exception as e:
        logger.warning("ingest %s failed: %s", ticker, e)
        return False


async def _ingest_nasdaq_ticker(ticker: str) -> bool:
    """Fetch latest daily bars for a NASDAQ ticker (no .VN suffix) and append to CSV."""
    try:
        import yfinance as yf
        import pandas as pd

        NASDAQ_DATA_DIR.mkdir(exist_ok=True)
        csv_path = NASDAQ_DATA_DIR / f"{ticker}_Historical.csv"

        if not csv_path.exists():
            # First run: bootstrap with 2 years of history
            raw = yf.download(ticker, period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning("ingest NASDAQ %s: no data from yfinance", ticker)
                return False
            raw = raw.reset_index()
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
            df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            df.to_csv(csv_path, index=False)
            logger.info("ingest NASDAQ %s: bootstrapped with %d rows", ticker, len(df))
            return True

        existing = pd.read_csv(csv_path)
        existing.columns = existing.columns.str.strip()
        last_date = pd.to_datetime(existing["Date"].iloc[-1])

        raw = yf.Ticker(ticker).history(period="1mo", interval="1d")
        if raw.empty:
            logger.info("ingest NASDAQ %s: no new data", ticker)
            return True

        raw = raw.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        new_rows = raw[raw["Date"] > last_date].copy()
        if new_rows.empty:
            logger.info("ingest NASDAQ %s: already up to date (%s)", ticker, last_date.date())
            return True

        new_rows["Date"] = new_rows["Date"].dt.strftime("%Y-%m-%d")
        latest = new_rows[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

        existing["Date"] = pd.to_datetime(existing["Date"]).dt.strftime("%Y-%m-%d")
        combined = (
            pd.concat([existing, latest])
            .drop_duplicates(subset="Date", keep="last")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        combined.to_csv(csv_path, index=False)
        logger.info("ingest NASDAQ %s: +%d new rows (up to %s)",
                    ticker, len(latest), latest["Date"].iloc[-1])
        return True
    except Exception as e:
        logger.warning("ingest NASDAQ %s failed: %s", ticker, e)
        return False


async def run_pipeline():
    """Main pipeline job — ingest → predict → store."""
    started = datetime.now(timezone.utc)
    ok, failed = [], []

    logger.info("Pipeline started — VN: %d tickers, NASDAQ: %d tickers",
                len(TICKERS), len(NASDAQ_TICKERS))

    # 1a. INGEST VN
    for ticker in TICKERS:
        success = await _ingest_ticker(ticker)
        if not success:
            failed.append(ticker)

    # 1b. INGEST NASDAQ
    for ticker in NASDAQ_TICKERS:
        await _ingest_nasdaq_ticker(ticker)   # failures logged but don't count as pipeline failure

    # 2 + 3. PREDICT (transform happens inside inference.py)
    tasks = ["task2_1", "task2_2_k3", "task2_2_k7", "task2_3_k3", "task2_3_k7"]
    col = predictions_col()

    for task in tasks:
        try:
            results = predict_all_tickers(task=task)
        except Exception as e:
            logger.error("predict_all_tickers(%s) failed: %s", task, e)
            continue

        # 4. STORE
        docs = []
        for r in results:
            if "error" in r:
                failed.append(r["ticker"])
                continue
            doc = {
                "ticker":     r["ticker"],
                "task":       r["task"],
                "run_at":     started,
                "last_close": r.get("last_close"),
                "predictions": r.get("predictions", {}),
                "unit":       r.get("unit", "VND"),
            }
            docs.append(doc)
            ok.append(r["ticker"])

        for doc in docs:
            await col.replace_one(
                {"ticker": doc["ticker"], "task": doc["task"]},
                doc,
                upsert=True,
            )

    completed = datetime.now(timezone.utc)
    duration  = (completed - started).total_seconds()

    await pipeline_runs_col().insert_one({
        "started_at":    started,
        "completed_at":  completed,
        "status":        "success" if not failed else ("partial" if ok else "failed"),
        "tickers_ok":    list(set(ok)),
        "tickers_failed": list(set(failed)),
        "duration_sec":  round(duration, 2),
    })

    logger.info(
        "Pipeline done in %.1fs — ok=%d failed=%d",
        duration, len(set(ok)), len(set(failed))
    )


def start_scheduler():
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(hour=6, minute=0, timezone="Asia/Ho_Chi_Minh"),
        id="daily_pipeline",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Scheduler started — pipeline fires daily at 06:00 ICT")
