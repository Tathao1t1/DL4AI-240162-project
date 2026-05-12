<div align="center">

# TensorFinance AI

### Deep Learning for Financial Time Series — CS313 Final Project

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)]()

An end-to-end platform for deep learning–based stock price forecasting and systematic trading, covering **31 NASDAQ tickers** (LSTM) and **28 Vietnam HOSE tickers** (CNN-LSTM) — from raw OHLCV data all the way to a live React dashboard served through a production FastAPI backend.

**[📺 Product Demo (Google Drive)](https://drive.google.com/drive/folders/1uLG1LHZHMZ8kd9zdVAq6DgSJLaYHC6DB)**
**[📺 Product Link(ngrok)](https://chooser-deprecate-briskly.ngrok-free.dev/)**
**[📺 Report for more details (Google Drive)](https://drive.google.com/drive/folders/1uLG1LHZHMZ8kd9zdVAq6DgSJLaYHC6DB)**



</div>

---

## 📋 Table of Contents

- [Product Demo](#-product-demo)
- [Project Tasks](#-project-tasks)
- [Model Performance](#-model-performance)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [License](#-license)

---

## 📺 Product Demo

A screen-recorded walkthrough of the live TensorFinance dashboard is available on Google Drive:

**[▶ Watch Demo](https://drive.google.com/drive/folders/1uLG1LHZHMZ8kd9zdVAq6DgSJLaYHC6DB)**

---

## 🎯 Project Tasks

This project is structured around five progressive tasks in the CS313 Deep Learning curriculum:

### Task 1 — LSTM Price Prediction (NASDAQ)

Individual LSTM models trained per ticker on 31 NASDAQ stocks. Three forecasting horizons:

| Sub-task | Description | Tickers |
|---|---|---|
| **1.1** | Predict the **next-day** closing price | 31 |
| **1.2** | Predict the closing price **k days ahead** (k=3, k=7) | 31 × 2 |
| **1.3** | Predict **k consecutive days** of closing prices (k=3, k=7) | 31 × 2 |

- Architecture: LSTM with 60-day lookback window
- Target: log-return → `price = last_close × exp(log_return)`
- Features: 20 ratio-normalized technical indicators (Returns, SMA/EMA, RSI, MACD, Bollinger Bands, ATR, Volume ratio)

### Task 2 — CNN-LSTM Price Prediction (Vietnam HOSE)

Transfer-learning approach across 28 Vietnam HOSE stocks:

| Sub-task | Description | Tickers |
|---|---|---|
| **2.1** | Predict the **next-day** closing price | 28 |
| **2.2** | Predict the **k-th day ahead** price (k=3, k=7) | 28 × 2 |
| **2.3** | Predict **k consecutive days** of closing prices (k=3, k=7) | 28 × 2 |

- Architecture: 1D CNN feature extractor → LSTM sequence model → per-ticker fine-tuned heads
- Shared backbone pre-trained on all 28 tickers captures cross-market patterns
- Features: 24 ratio-normalized indicators including VN-specific candlestick features

### Task 3 — Trading Signal Generation

Binary classifiers generating **Buy / Sell / Hold** signals per ticker:

- Separate buy-signal model and sell-signal model per ticker
- CNN-LSTM architecture reused from Task 2 backbone
- Decision threshold tuned per ticker to optimize F1 on validation set
- Output: signal label + confidence score

### Task 4 — Portfolio Optimization

Two allocation strategies derived from Task 2/3 metrics:

| Strategy | Objective |
|---|---|
| **Prudent** | Minimize risk — weighted by inverse volatility and drawdown scores |
| **Risk-taking** | Maximize return potential — weighted by profitability ranking |

### Task 5 — Production Deployment

Full-stack deployment serving all model outputs via REST API and a live dashboard:

- FastAPI backend with JWT authentication
- APScheduler daily pipeline: ingest → feature engineering → predict → MongoDB upsert
- SSE real-time price streaming via yfinance
- React 19 + TypeScript dashboard with candlestick charts, signal view, and portfolio view
- Docker Compose with Nginx reverse proxy

---

## 📊 Model Performance

All metrics are computed on a **held-out test set** using strict chronological train/validation/test splits. No data leakage — scalers are fitted on training data only.

### Task 1 — NASDAQ LSTM (averaged over 31 tickers)

| Horizon | MAE (USD) | MAPE |
|---|---|---|
| Next-day (k=1) | 1.25 | 1.65% |
| k=3 | 2.23 | 2.91% |
| k=7 | 3.48 | 4.57% |

### Task 2 — Vietnam CNN-LSTM (averaged over 28 tickers, prices in VND)

| Horizon | MAE (VND) | RMSE (VND) |
|---|---|---|---|
| Next-day (k=1) | 474 | 729 |
| k=3 | 845 | 1,271 |
| k=7 | 1,306 | 1,916 | 

### Task 3 — Trading Signal Classifiers (averaged over 28 tickers)

| Signal | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Buy | 0.781 | 0.597 | 0.487 | 0.793 |
| Sell | 0.806 | 0.598 | 0.499 | 0.774 |

---

## 🏗 System Architecture

```
  ┌────────────────────────────────────────────────────────┐
  │                    React Dashboard                     │
  │  Price Charts · Trading Signals · Portfolio · Live SSE │
  └──────────────────────┬─────────────────────────────────┘
                         │  REST + SSE
  ┌──────────────────────▼─────────────────────────────────┐
  │                   FastAPI  /api/v1                     │
  │  /predict  /signals  /portfolio  /live  /market  /auth │
  └──────────┬──────────────────────────┬──────────────────┘
             │                          │
  ┌──────────▼──────────┐  ┌────────────▼──────────────────┐
  │   TF/Keras Models   │  │           MongoDB             │
  │  Task 1 · 2 · 3 · 4 │  │  predictions · pipeline_runs  │
  │  (in-memory cache)  │  │  users                        │
  └─────────────────────┘  └───────────────────────────────┘
             ▲
  ┌──────────┴──────────┐
  │  APScheduler 06:00  │  Daily: ingest → feature eng → predict → upsert
  │  ICT (Asia/HCM)     │
  └─────────────────────┘
```

**Data flow:**
```
Raw OHLCV CSVs
  → feature engineering (ratio-normalized indicators)
  → StandardScaler (fitted on train split only)
  → LSTM / CNN-LSTM model
  → log-return prediction → exp() reconstruction → price
  → MongoDB upsert (daily at 06:00 ICT)
  → FastAPI serves cached predictions or live inference
  → React dashboard
```

---

## 🚀 Quick Start

### Visting the website for quick view : **[📺 Product Link(ngrok)](https://chooser-deprecate-briskly.ngrok-free.dev/)**

### Option A — Docker 

**Prerequisites:** [Docker Desktop](https://docs.docker.com/desktop/)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd DL4AI-240162-project

# 2. Configure environment
cp .env.example .env
# Edit .env — set MONGO_URL, MONGO_DB, JWT_SECRET

# 3. Start all services (API + UI + MongoDB)
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

- Dashboard: **http://localhost**
- API docs (Swagger): **http://localhost/api/docs**

> For production (external MongoDB Atlas), use `docker compose up --build` without the local override and point `MONGO_URL` to your Atlas connection string.

---

### Option B — Local Development

**Prerequisites:** Python 3.11, Node.js 20+, MongoDB 7 running on `localhost:27017`

**Backend:**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # fill in MONGO_URL, MONGO_DB, JWT_SECRET
uvicorn api.main:app --reload --port 8000
# API: http://localhost:8000 | Swagger: http://localhost:8000/docs
```

**Frontend** (separate terminal):

```bash
cd ui
npm install
npm run dev   # → http://localhost:5173
```

**`.env` minimum:**
```env
MONGO_URL=mongodb://localhost:27017
MONGO_DB=tensorfinance
JWT_SECRET=any-hex-string-at-least-32-chars
```

---

### Option C — Notebooks (model training)

```bash
pip install -r requirements.txt
jupyter lab
```

| Notebook | Purpose |
|---|---|
| `240162-project-notebook.ipynb` | Main training notebook — Tasks 1–4 |
| `baseline_comparison.ipynb` | Baseline model comparison |
| `demo_LSTM_CNN.ipynb` | Architecture walkthrough |

---

### Running Tests

```bash
pytest tests/ -v

# Single test
pytest tests/test_preprocessing.py::test_price_reconstruction_uses_exp -v
```

---

## 📁 Project Structure

```
DL4AI-240162-project/
│
├── api/                          # FastAPI backend
│   ├── main.py                   # App factory, lifespan, CORS
│   ├── scheduler.py              # Daily pipeline (APScheduler)
│   ├── database.py               # Motor async MongoDB client
│   ├                   
│   └── routers/
│       ├── predict.py            # Task 2 — VN prediction endpoints
│       ├── predict_nasdaq.py     # Task 1 — NASDAQ prediction endpoints
│       ├── signals.py            # Task 3 — buy/sell signal endpoints
│       ├── portfolio.py          # Task 4 — portfolio endpoints
│       ├── market.py             # Quote, OHLCV history, RSI
│       └── live.py               # SSE real-time price stream
│
├── models/
│   ├── task1_1/                  # NASDAQ next-day LSTM (31 tickers)
│   ├── task1_2/k3, k7/           # NASDAQ k-th day LSTM
│   ├── task1_3/k3, k7/           # NASDAQ k consecutive days LSTM
│   ├── task2/                    # VN CNN-LSTM + per-ticker scalers
│   ├── task3/                    # Buy/sell classifiers + metrics CSVs
│   ├── task4/                    # Portfolio JSONs + risk/profitability CSVs
│   └── predictions/              # Cached test-set predictions (CSVs)
│
├── clean-historical-data-2026/   # VN HOSE OHLCV CSVs (28 tickers)
├── nasdaq-historical-data/       # NASDAQ OHLCV CSVs (31 tickers, auto-managed)
├── inference.py                  # VN feature engineering + Task 2 inference entry point
│
├── ui/                           # React 19 + TypeScript dashboard
│   └── src/
│       ├── App.tsx               # Root shell with market/ticker state
│       ├── components/           # PricePredictions, Sidebar, AuthModal, ...
│       └── services/
│           └── marketService.ts  # All API call wrappers
│
├── tests/                        # Pytest test suite
├── 240162-project-notebook.ipynb # Main training notebook
├── docker-compose.yml            # Production services (api + ui)
├── docker-compose.local.yml      # Local override (adds mongo container)
├── Dockerfile.api
├── requirements.txt
└── .env.example
```

---

## 📡 API Reference

Interactive docs available at `/docs` (Swagger UI) or `/redoc` when the server is running.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/predict/{ticker}?task=task2_1` | VN price prediction (next-day, k=3, k=7) |
| `GET` | `/api/v1/predict/nasdaq/{ticker}?k=1` | NASDAQ price prediction |
| `GET` | `/api/v1/predict/nasdaq/consecutive/{ticker}?k=3` | NASDAQ k consecutive days |
| `GET` | `/api/v1/signals/{ticker}` | Buy/sell signal + confidence score |
| `GET` | `/api/v1/signals/all/latest` | Signals for all 28 VN tickers |
| `GET` | `/api/v1/portfolio/summary` | Both portfolio allocations |
| `GET` | `/api/v1/portfolio/risk-scores` | Per-ticker risk ranking |
| `GET` | `/api/v1/market/history/{ticker}?period=1mo&market=VN` | OHLCV history |
| `GET` | `/api/v1/market/quote/{ticker}` | Current quote |
| `GET` | `/api/v1/live/prices?tickers=FPT&market=VN` | SSE live price stream |
| `GET` | `/api/v1/pipeline/status` | Last pipeline run status |
| `POST` | `/api/v1/pipeline/trigger` | Manually trigger data pipeline (auth required) |


---

## 📜 License

Developed as the final project for **CS313 — Deep Learning**, Fulbright University Vietnam.
