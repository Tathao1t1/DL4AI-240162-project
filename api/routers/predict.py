"""
/api/v1/predict endpoints — Task 2 price predictions.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from inference import predict_price, predict_all_tickers, TICKERS
from api.database import predictions_col

router = APIRouter(prefix="/predict", tags=["predict"])

VALID_TASKS = ["task2_1", "task2_2_k3", "task2_2_k7"]


@router.get("/tickers")
async def list_tickers():
    """Return the list of supported VN tickers."""
    return {"tickers": TICKERS, "count": len(TICKERS)}


@router.get("/{ticker}")
async def predict_ticker(
    ticker: str,
    task: str = Query("task2_2_k7", description="task2_1 | task2_2_k3 | task2_2_k7"),
    use_cache: bool = Query(True, description="Return latest stored result if available"),
):
    """
    Return price prediction for a single ticker.
    use_cache=true (default): return latest MongoDB result (fast).
    use_cache=false: run live inference (slow, ~1–3 s).
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker '{ticker}' not in supported universe")
    if task not in VALID_TASKS:
        raise HTTPException(400, f"task must be one of {VALID_TASKS}")

    if use_cache:
        try:
            col = predictions_col()
            doc = await col.find_one(
                {"ticker": ticker, "task": task},
                sort=[("run_at", -1)],
            )
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception:
            pass  # MongoDB unavailable — fall through to live inference

    # Live inference fallback
    try:
        result = predict_price(ticker, task=task)
    except Exception as e:
        raise HTTPException(500, str(e))

    result["run_at"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("/all/latest")
async def predict_all(
    task: str = Query("task2_2_k7"),
):
    """
    Return the latest stored prediction for every ticker.
    Falls back to live inference for any ticker with no stored result.
    """
    if task not in VALID_TASKS:
        raise HTTPException(400, f"task must be one of {VALID_TASKS}")

    col = predictions_col()
    # One aggregation query — latest doc per ticker for this task
    pipeline = [
        {"$match": {"task": task}},
        {"$sort": {"run_at": -1}},
        {"$group": {"_id": "$ticker", "doc": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$doc"}},
    ]
    cached = {d["ticker"]: d async for d in col.aggregate(pipeline)
              if await _strip_id(d)}

    results = []
    for ticker in TICKERS:
        if ticker in cached:
            results.append(cached[ticker])
        else:
            try:
                r = predict_price(ticker, task=task)
                r["run_at"] = datetime.now(timezone.utc).isoformat()
                results.append(r)
            except Exception as e:
                results.append({"ticker": ticker, "task": task, "error": str(e)})

    return {"task": task, "count": len(results), "results": results}


@router.get("/{ticker}/history")
async def prediction_history(
    ticker: str,
    task: str = Query("task2_2_k7"),
    limit: int = Query(30, le=100),
):
    """Last N stored predictions for one ticker (for trend charts)."""
    ticker = ticker.upper()
    col = predictions_col()
    cursor = col.find(
        {"ticker": ticker, "task": task},
        {"_id": 0},
        sort=[("run_at", -1)],
        limit=limit,
    )
    docs = [d async for d in cursor]
    docs.reverse()  # chronological order
    return {"ticker": ticker, "task": task, "history": docs}


async def _strip_id(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc
