"""
/api/v1/portfolio endpoints — Task 4 portfolio data.
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
import pandas as pd

ROOT     = Path(__file__).parent.parent.parent
MANIFEST = ROOT / "models" / "task4" / "portfolio_manifest.json"

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


@router.get("/prudent")
async def get_prudent():
    """Return the Prudent (risk-averse) portfolio."""
    m = _load_manifest()
    with open(ROOT / "models" / "task4" / "portfolio_prudent.json") as f:
        return json.load(f)


@router.get("/risk-taking")
async def get_risk_taking():
    """Return the Risk-Taking portfolio."""
    with open(ROOT / "models" / "task4" / "portfolio_risk_taking.json") as f:
        return json.load(f)


@router.get("/risk-scores")
async def get_risk_scores():
    """Return per-ticker risk scores (all 27 tickers)."""
    path = ROOT / "models" / "task4" / "risk_scores.csv"
    if not path.exists():
        raise HTTPException(404, "risk_scores.csv not found")
    df = pd.read_csv(path)
    return {"tickers": df.to_dict(orient="records")}


@router.get("/profitability")
async def get_profitability():
    """Return profitability scores from Task 4.1 candidate selection."""
    path = ROOT / "models" / "task4" / "profitability_scores.csv"
    if not path.exists():
        raise HTTPException(404, "profitability_scores.csv not found")
    df = pd.read_csv(path)
    return {"candidates": df.to_dict(orient="records")}


@router.get("/summary")
async def get_summary():
    """Return both portfolios side by side for comparison."""
    with open(ROOT / "models" / "task4" / "portfolio_prudent.json") as f:
        prudent = json.load(f)
    with open(ROOT / "models" / "task4" / "portfolio_risk_taking.json") as f:
        risk_taking = json.load(f)
    return {"prudent": prudent, "risk_taking": risk_taking}
