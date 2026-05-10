# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TensorFinance** — CS313 Deep Learning final project. End-to-end stock price forecasting and systematic trading platform covering NASDAQ (Task 1, LSTM) and Vietnam HOSE (Task 2–4, CNN-LSTM). Deployed as a FastAPI REST API + React dashboard + MongoDB, orchestrated via Docker Compose.

## Commands

### Backend (Python 3.11)
```bash
# Install deps
pip install -r requirements.txt
pip install fastapi==0.115.0 uvicorn[standard]==0.32.0 motor==3.6.0 apscheduler==3.10.4 yfinance==0.2.48

# Run API (local, requires MongoDB on localhost:27017)
cp .env.example .env   # fill in MONGO_URL, MONGO_DB, JWT_SECRET
uvicorn api.main:app --reload --port 8000

# Run inference self-test (no API needed)
python inference.py

# Run tests
pytest tests/ -v
# Run a single test
pytest tests/test_preprocessing.py::test_price_reconstruction_uses_exp -v
```

### Frontend (Node 20+)
```bash
cd ui
npm install
npm run dev      # → http://localhost:5173
npm run build    # TypeScript check + Vite production bundle
```

### Docker (production)
```bash
cp .env.example .env   # set MONGO_USER, MONGO_PASSWORD, JWT_SECRET
docker-compose up --build
# App: http://localhost  |  API docs: http://localhost/api/docs
```

### Notebooks (model training)
```bash
jupyter lab
# Main notebook: 240162-project-notebook.ipynb  (Tasks 1–4 training)
# Task 1 improved: task1_improved_model.ipynb
# Baselines:       baseline_comparison.ipynb
```

## Architecture

### Data flow
```
Raw OHLCV CSVs
  → feature engineering (_add_vn_features / _add_nasdaq_features)
  → StandardScaler (fitted on training split only)
  → LSTM / CNN-LSTM model
  → log-return prediction → exp() reconstruction → price
  → MongoDB upsert (via APScheduler daily at 06:00 ICT)
  → FastAPI serves cached predictions or runs live inference
  → React dashboard
```

### Directory layout (key paths)
- `inference.py` — Task 2 feature engineering + VN inference entry point. Also called by the scheduler. The `_add_vn_features()` function here **must stay in sync** with the training notebook cell `ea04e6ec`.
- `api/main.py` — FastAPI app, lifespan startup (model preload + scheduler + MongoDB indexes).
- `api/scheduler.py` — APScheduler daily pipeline: ingest yfinance → feature eng → predict all → MongoDB upsert.
- `api/database.py` — Motor async MongoDB client, three collections: `predictions`, `pipeline_runs`, `users`.
- `api/routers/predict.py` — VN price prediction endpoints (Task 2); cache-first, live-inference fallback.
- `api/routers/predict_nasdaq.py` — NASDAQ endpoints (Tasks 1.1–1.3); in-memory model cache keyed `(ticker, k)`.
- `api/routers/signals.py` — Task 3 buy/sell signal classifiers.
- `api/routers/portfolio.py` — Task 4 portfolio composition endpoints.
- `api/routers/live.py` — SSE real-time price stream via yfinance.
- `api/utils/validation.py` — `validate_ohlcv()` shared by both VN and NASDAQ loaders.
- `models/` — All trained artifacts (never retrained by the API):
  - `task1_1/next_day/per_ticker/{TICKER}/` — NASDAQ next-day LSTM (`model.keras`, `scaler_X.pkl`, `scaler_y.pkl`, `metadata.json`)
  - `task1_2/k3|k7/per_ticker/{TICKER}/` — NASDAQ k-th day
  - `task1_3/k3|k7/per_ticker/{TICKER}/` — NASDAQ k consecutive days
  - `task2/` — VN CNN-LSTM backbone + per-ticker scalers + `model_manifest.json`
  - `task3/` — Per-ticker buy/sell classifiers (`task3_buy_{TICKER}.keras`, `task3_sell_{TICKER}.keras`)
  - `task4/` — `portfolio_manifest.json`, risk/profitability CSVs, two allocation JSONs
- `clean-historical-data-2026/` — VN OHLCV CSVs (`{TICKER}_Historical.csv`). Row 0 = header, row 1 = ticker label (skiprows=[1] on read).
- `nasdaq-historical-data/` — NASDAQ OHLCV CSVs (auto-bootstrapped by scheduler on first request).
- `ui/src/` — React 19 + TypeScript dashboard. `App.tsx` = shell with market/ticker state. `services/marketService.ts` = all API calls.

### Model design invariants
- **All features are ratio-based** (not absolute values) — scale-invariant across tickers and time.
- **Target variable is log-return**, not raw price. Reconstruction: `price = last_close * exp(cumulative_log_return)`. Never use the linear `1 + x` approximation.
- **scaler_y is fitted only on training rows** (rows `[lookback : n_train + lookback]`), never on pre-window rows. This is tested in `tests/test_preprocessing.py`.
- **Lookback window**: 60 days for NASDAQ (Task 1), 30 days for VN (Task 2). These are baked into saved models; do not change without retraining.
- **No data leakage**: train → validation → test is strictly chronological; scalers see only training data.

### VN vs NASDAQ feature differences
- VN features (24 total, `inference.py::_add_vn_features`): includes VN-specific candlestick features (`VN_limit_prox`, `OC_pct`, `Upper_shadow`, `Lower_shadow`).
- NASDAQ features (20 total, `predict_nasdaq.py::_add_nasdaq_features`): subset without VN-specific candlestick features.

### Inference caching
- VN models: `_model_cache` and `_scaler_cache` dicts in `inference.py`, keyed by file path string.
- NASDAQ models: `_cache` dict in `predict_nasdaq.py`, keyed `(ticker, k)` for task1_1/1_2; `_cache_consecutive` for task1_3.
- Both are warmed at startup via `preload_all()` called in `lifespan`.

### API prefix
All endpoints live under `/api/v1`. Swagger UI: `GET /docs`.

### Auth
JWT-based (`api/auth/`). The React dashboard requires login; the `AuthModal` gates the entire UI shell.

### Environment variables
```
MONGO_URL=mongodb://localhost:27017   # or mongodb://{user}:{pass}@mongo:27017 in Docker
MONGO_DB=tensorfinance
JWT_SECRET=<hex string>
# Docker Compose also needs:
MONGO_USER=admin
MONGO_PASSWORD=...
```
