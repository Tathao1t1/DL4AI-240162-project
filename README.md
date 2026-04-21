<div align="center">

# QuantPulse AI

### Financial Time Series Prediction & Systematic Trading Research Platform

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.13-D00000?style=flat-square&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![MongoDB](https://img.shields.io/badge/MongoDB-7-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

An end-to-end research and deployment platform for deep learning-based financial forecasting — covering **feature engineering**, **multi-horizon price prediction**, **systematic trading signal generation**, and **portfolio optimization** across Vietnamese (HOSE) and NASDAQ equity markets.

</div>

---

## Research Overview

This project addresses the full ML research lifecycle applied to financial time series:

```
Raw OHLCV Data
      │
      ▼
Feature Engineering ──────── 20–24 ratio-based technical indicators
      │                       (Returns, SMA/EMA, RSI, MACD, BB, ATR, Volatility)
      ▼
Model Training ───────────── LSTM (NASDAQ) · CNN-LSTM (Vietnam)
      │                       Per-ticker models · Lookback window = 60 days
      ▼
Evaluation & Backtesting ─── MAE · MAPE · RMSE on held-out test set
      │                       Train / Validation / Test split per ticker
      ▼
Production Pipeline ─────── Daily auto-ingest → inference → MongoDB store
      │                       FastAPI serving · React dashboard · Live SSE feed
      ▼
Strategy Layer ──────────── Buy/Sell signal classifiers · Portfolio optimization
```

---

## Key Technical Highlights

### Feature Engineering for Financial Time Series
Hand-crafted **20–24 ratio-normalized indicators** designed to be scale-invariant across tickers and time periods:

| Category | Features |
|----------|----------|
| **Momentum** | Returns over 1d / 5d / 10d / 20d windows |
| **Trend** | SMA10 vs SMA20 · SMA20 vs SMA50 · EMA10 vs EMA20 · Price vs SMA20/50 |
| **Oscillators** | RSI-14 · MACD (line, signal, histogram) — all normalized by price |
| **Volatility** | Bollinger Band width & %B · ATR-14 (% of price) · Rolling std (10d, 20d) |
| **Volume** | Volume ratio vs 20-day average |
| **Range** | High-Low range as % of close |

All features expressed as **ratios** (not absolute values) to ensure stationarity and cross-ticker comparability.

---

### Deep Learning Architectures

#### Task 1 — Per-Ticker LSTM (NASDAQ)
- Separate LSTM model per ticker trained on 2 years of daily data
- Lookback window: 60 trading days → single-step output (k-th day price)
- Target: k-th day closing price via return prediction → inverse-transformed to price
- Trained for three horizons independently: `k=1`, `k=3`, `k=7`

#### Task 2 — CNN-LSTM (Vietnam)
- **Global backbone** shared across all 28 VN tickers — captures market-wide patterns
- **Per-ticker heads** — fine-tune to individual stock behavior
- 1D CNN extracts local temporal features → LSTM models sequential dependencies
- Trained for three horizons: next-day · +3rd day · +7th day

#### Task 3 — Trading Signal Classifiers
- Separate **binary classifiers** (buy signal model + sell signal model) per ticker
- Predicts probability of a significant upward/downward move
- Buy/Sell/Hold decision via confidence threshold
- Evaluated on AUC, F1, Precision, Recall

#### Task 4 — Portfolio Optimization
- Risk scoring: per-ticker volatility and drawdown metrics
- Profitability scoring: return potential ranking
- Output: two allocation strategies — **prudent** (risk-minimizing) and **risk-taking** (return-maximizing)

---

### Model Evaluation

Each model is evaluated on a strict **train / validation / test chronological split** (no data leakage):

| Task | Metric | Notes |
|------|--------|-------|
| Task 1 (NASDAQ) | MAE · MAPE | Per-ticker, reported in USD |
| Task 2 (Vietnam) | MAE · MAPE · RMSE | Per-ticker, reported in VND |
| Task 3 (Signals) | AUC · F1 · Precision · Recall | Stored in `task3_buy/sell_metrics.csv` |

Model artifacts include `scaler_X.pkl` and `scaler_y.pkl` per ticker, fitted only on training data and applied consistently at inference.

---

### Production ML Pipeline

```
06:00 ICT (daily, APScheduler)
    │
    ├── Ingest VN      yfinance (.VN) → append new rows to local OHLCV CSVs
    ├── Ingest NASDAQ  yfinance (plain) → append new rows to local OHLCV CSVs
    ├── Feature Eng.   Recompute all indicators on updated data (inside inference.py)
    ├── Predict        Task 2 models → predictions for all 3 horizons × 28 tickers
    └── Store          Upsert results into MongoDB predictions collection
```

- Live top-up on every prediction request (no stale data)
- Model cache in-memory — O(1) inference after first load
- Market-aware SSE stream (VN tries `.VN` suffix; NASDAQ skips it)
- Manual pipeline trigger: `POST /api/v1/pipeline/run`

---

## System Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                     React Dashboard                     │
  │  Price Predictions · Trading Signals · Portfolio View   │
  │  Candlestick Charts · Live SSE Prices · Ticker Search   │
  └──────────────────────┬──────────────────────────────────┘
                         │ REST + SSE
  ┌──────────────────────▼──────────────────────────────────┐
  │                    FastAPI Backend                      │
  │  /predict  /signals  /portfolio  /live  /market  /auth  │
  └───────────┬───────────────────────────┬─────────────────┘
              │                           │
  ┌───────────▼──────────┐   ┌────────────▼────────────────┐
  │   TF/Keras Models    │   │         MongoDB             │
  │  Task1 · 2 · 3 · 4  │   │  predictions · users ·      │
  │  (in-memory cache)   │   │  pipeline_runs              │
  └──────────────────────┘   └─────────────────────────────┘
```

---

## Tech Stack

<table>
<tr>
<td valign="top" width="50%">

**Research & Modeling**
- **TensorFlow 2.21 / Keras 3.13** — LSTM, CNN-LSTM
- **scikit-learn 1.8** — StandardScaler, train/test split, metrics
- **NumPy · Pandas** — feature engineering, time series manipulation
- **yfinance** — OHLCV data ingestion

</td>
<td valign="top" width="50%">

**Engineering & Deployment**
- **FastAPI** — async REST API + SSE streaming
- **Motor** — async MongoDB driver
- **APScheduler** — daily pipeline automation
- **Docker Compose + Nginx** — containerized deployment
- **React 19 + TypeScript + Vite** — research dashboard

</td>
</tr>
</table>

---

## Project Structure

```
DL4AI-240162-project/
│
├── api/                          # Production API layer
│   ├── scheduler.py              # Daily data pipeline (06:00 ICT cron)
│   ├── auth/                     # JWT auth
│   └── routers/
│       ├── predict.py            # Task 2 — VN inference endpoint
│       ├── predict_nasdaq.py     # Task 1 — NASDAQ inference endpoint
│       ├── signals.py            # Task 3 — signal endpoint
│       ├── portfolio.py          # Task 4 — portfolio endpoint
│       ├── market.py             # Quote, OHLCV history, RSI
│       └── live.py               # SSE real-time price stream
│
├── models/
│   ├── task1_1/                  # NASDAQ LSTM — next-day (31 tickers)
│   ├── task1_2/k3, k7/           # NASDAQ LSTM — +3d/+7d (31 tickers each)
│   ├── task2/                    # VN CNN-LSTM models + per-ticker scalers
│   ├── task3/                    # Buy/sell classifiers + evaluation metrics
│   └── task4/                    # Portfolio JSONs + risk/profitability scores
│
├── clean-historical-data-2026/   # VN HOSE OHLCV CSVs (28 tickers)
├── nasdaq-historical-data/       # NASDAQ OHLCV CSVs (auto-managed, 31 tickers)
├── inference.py                  # Feature engineering + Task 2 inference
│
├── ui/                           # Research dashboard (React + TypeScript)
├── docker-compose.yml
├── requirements.txt
└── 240162-project-notebook.ipynb # Model training & experiments
```

---

## Getting Started

### Option A — Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

App on **http://localhost** · API docs on **http://localhost/api/docs**

### Option B — Local Development

**Prerequisites:** Python 3.11 · Node.js 20+ · MongoDB 7 on `localhost:27017`

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd ui && npm install && npm run dev   # → http://localhost:5173
```

**`.env`**
```env
MONGO_URL=mongodb://localhost:27017
MONGO_DB=quantpulse
JWT_SECRET=change-me-in-production
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/predict/{ticker}?task=task2_1` | VN price prediction |
| `GET` | `/api/v1/predict/nasdaq/{ticker}?k=1\|3\|7` | NASDAQ price prediction |
| `GET` | `/api/v1/signals/{ticker}` | Buy/sell signal + confidence |
| `GET` | `/api/v1/portfolio/summary` | Portfolio allocations |
| `GET` | `/api/v1/market/history/{ticker}?period=1mo&market=VN` | OHLCV bars |
| `GET` | `/api/v1/live/prices?tickers=FPT&market=VN` | SSE live price stream |
| `POST` | `/api/v1/pipeline/run` | Manually trigger pipeline |

---

## License

Developed as the final project for **CS313 — Deep Learning** course.
