# QuantPulse AI — Stock Market Prediction Platform

A full-stack deep learning web application for stock price prediction, trading signal generation, and portfolio optimization across **Vietnamese (HOSE)** and **NASDAQ** equity markets.

---

## Overview

QuantPulse AI combines per-ticker LSTM/CNN models trained on historical OHLCV data with a production-ready web dashboard. Users can explore multi-horizon price forecasts, buy/sell signals, and optimized portfolio allocations — all updating automatically via a daily data pipeline.

---

## Features

| Feature | Details |
|---------|---------|
| **Price Prediction** | Next-day, 3rd-day, and 7th-day closing price forecasts |
| **Trading Signals** | Buy / Sell / Hold with model confidence scores |
| **Portfolio Optimization** | Prudent and risk-taking allocations across VN tickers |
| **Live Prices** | Real-time quote updates via Server-Sent Events (SSE) |
| **Candlestick Charts** | OHLCV visualization with TradingView Lightweight Charts |
| **Multi-market Search** | Fuzzy ticker search across VN and NASDAQ |
| **Auth** | JWT-based user accounts with localStorage token persistence |
| **Daily Pipeline** | Auto-ingest + re-predict at 06:00 ICT via APScheduler |

---

## Machine Learning Tasks

### Task 1 — NASDAQ Price Prediction (LSTM)
- **Architecture:** Per-ticker LSTM with 20 ratio-based technical indicators
- **Horizons:** Next-day (`k=1`), 3rd-day (`k=3`), 7th-day (`k=7`)
- **Tickers:** 31 NASDAQ stocks (AAPL, INTC, TXN, AMGN, CBSH, MAT, MNST, ROST, …)
- **Features:** Return (1/5/10/20d), SMA/EMA ratios, RSI-14, MACD, Bollinger Bands, ATR, volatility, volume ratio

### Task 2 — Vietnam Price Prediction (CNN-LSTM)
- **Architecture:** Global CNN-LSTM backbone + per-ticker heads
- **Horizons:** Next-day (`task2_1`), 3rd-day (`task2_2_k3`), 7th-day (`task2_2_k7`)
- **Tickers:** 28 HOSE stocks (FPT, ACB, VNM, DHG, FMC, GMD, STB, SSI, …)
- **Features:** 24 engineered indicators derived from OHLCV data

### Task 3 — Trading Signal Classification
- **Architecture:** Binary classifiers (buy model + sell model per ticker)
- **Output:** Buy / Sell probability + Hold threshold
- **Models:** 54 Keras models (27 buy + 27 sell)
- **Metrics:** AUC, F1, Precision, Recall per ticker

### Task 4 — Portfolio Optimization
- **Output:** Two recommended portfolios — *prudent* (risk-averse) and *risk-taking* (return-maximizing)
- **Method:** Risk scoring + profitability scoring across 27 VN tickers
- **Artifacts:** `portfolio_prudent.json`, `portfolio_risk_taking.json`, per-ticker risk/profitability scores

---

## Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — async REST API
- [TensorFlow 2 / Keras 3](https://keras.io/) — model inference
- [Motor](https://motor.readthedocs.io/) — async MongoDB driver
- [APScheduler](https://apscheduler.readthedocs.io/) — daily pipeline cron
- [yfinance](https://github.com/ranaroussi/yfinance) — market data ingestion
- [python-jose](https://github.com/mpdavis/python-jose) + [bcrypt](https://github.com/pyca/bcrypt/) — JWT auth

**Frontend**
- [React 19](https://react.dev/) + TypeScript + [Vite 6](https://vite.dev/)
- [Tailwind CSS 4](https://tailwindcss.com/) — styling
- [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) — candlestick charts
- [Recharts](https://recharts.org/) — area/line charts
- [Fuse.js](https://www.fusejs.io/) — client-side fuzzy search
- [Motion](https://motion.dev/) — animations

**Infrastructure**
- MongoDB 7 — predictions cache + pipeline run logs
- Docker + Docker Compose — containerized deployment
- Nginx — reverse proxy + static file serving

---

## Project Structure

```
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry point & router registration
│   ├── database.py             # Motor/MongoDB connection
│   ├── scheduler.py            # Daily ingest → predict → store pipeline
│   ├── auth/                   # JWT register/login/me
│   └── routers/
│       ├── predict.py          # Task 2 VN price predictions
│       ├── predict_nasdaq.py   # Task 1 NASDAQ price predictions
│       ├── signals.py          # Task 3 buy/sell signals
│       ├── portfolio.py        # Task 4 portfolio endpoints
│       ├── market.py           # Quote, history, RSI endpoints
│       ├── live.py             # SSE live price stream
│       └── search.py           # Ticker fuzzy search
│
├── ui/                         # React TypeScript frontend
│   └── src/
│       ├── components/         # PricePredictions, TradingSignals, Portfolio, …
│       ├── services/           # API client (marketService, authService)
│       ├── contexts/           # AuthContext
│       └── hooks/              # useLivePrices (SSE)
│
├── models/
│   ├── task1_1/                # NASDAQ next-day models (per ticker)
│   ├── task1_2/k3, k7/        # NASDAQ 3rd/7th-day models (per ticker)
│   ├── task2/                  # VN CNN-LSTM models + scalers
│   ├── task3/                  # VN buy/sell signal models
│   └── task4/                  # Portfolio JSON + score CSVs
│
├── clean-historical-data-2026/ # VN HOSE OHLCV CSVs (28 tickers)
├── nasdaq-historical-data/     # NASDAQ OHLCV CSVs (auto-managed)
├── inference.py                # Task 2 inference helper
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── nginx.conf
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Option A — Docker Compose (Recommended)

```bash
cp .env.example .env
docker-compose up --build
```

Opens on **http://localhost** (Nginx proxies UI + API).

### Option B — Local Development

**Prerequisites:** Python 3.11, Node.js 20+, MongoDB 7 running on `localhost:27017`

**1. Backend**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
uvicorn api.main:app --reload --port 8000
```

**2. Frontend** (new terminal)
```bash
cd ui
npm install
npm run dev                      # http://localhost:5173
```

**3. Environment variables** (`.env`)
```
MONGO_URL=mongodb://localhost:27017
MONGO_DB=quantpulse
JWT_SECRET=change-me-in-production
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/market/quote/{ticker}?market=VN\|NASDAQ` | Current quote |
| `GET` | `/api/v1/market/history/{ticker}?period=1mo&market=VN` | OHLCV bars |
| `GET` | `/api/v1/predict/{ticker}?task=task2_1` | VN price prediction |
| `GET` | `/api/v1/predict/nasdaq/{ticker}?k=1\|3\|7` | NASDAQ price prediction |
| `GET` | `/api/v1/signals/{ticker}` | Buy/sell signal |
| `GET` | `/api/v1/portfolio/summary` | Portfolio allocations |
| `GET` | `/api/v1/live/prices?tickers=FPT&market=VN` | SSE live price stream |
| `GET` | `/api/v1/search?q=fpt` | Ticker search |
| `POST` | `/api/v1/auth/register` | Create account |
| `POST` | `/api/v1/auth/login` | Login → JWT token |
| `POST` | `/api/v1/pipeline/run` | Manually trigger pipeline |

Interactive docs: **http://localhost:8000/docs**

---

## Data Pipeline

The daily pipeline runs automatically at **06:00 ICT**:

1. **Ingest VN** — fetch latest OHLCV via yfinance (`.VN` suffix) → append to CSVs
2. **Ingest NASDAQ** — fetch latest OHLCV via yfinance → append to CSVs
3. **Predict** — run Task 2 inference for all 3 horizons across all VN tickers
4. **Store** — upsert results into MongoDB `predictions` collection

Manual trigger: `POST /api/v1/pipeline/run`

---

## License

This project was developed as part of the **CS313 Deep Learning** course final project.
