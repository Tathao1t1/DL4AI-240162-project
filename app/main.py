"""
main.py — FastAPI application for DL4AI stock price prediction.

Run:
    cd /Users/ttt/Downloads/DL4AI-240162-project
    .venv/bin/uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /                          health check
    GET  /tickers                   list available tickers
    GET  /predict/1.1/{ticker}      next-day price prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predict_next_day, list_available_tickers

app = FastAPI(
    title       = 'DL4AI Stock Price Prediction API',
    description = 'LSTM-based next-day close price prediction for NASDAQ stocks.',
    version     = '1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ['*'],
    allow_methods  = ['*'],
    allow_headers  = ['*'],
)


@app.get('/')
def health():
    return {'status': 'ok', 'service': 'DL4AI Prediction API'}


@app.get('/tickers')
def get_tickers(task: str = '1.1'):
    """List all tickers with a saved model for the given task."""
    tickers = list_available_tickers(task)
    if not tickers:
        raise HTTPException(status_code=404, detail=f'No models found for task {task}.')
    return {'task': task, 'count': len(tickers), 'tickers': tickers}


@app.get('/predict/1.1/{ticker}')
def predict_1_1(ticker: str):
    """
    Next-day close price prediction for a NASDAQ ticker (Task 1.1).

    Fetches the most recent market data from yfinance, applies the saved
    LSTM model, and returns the predicted price with a 90% confidence interval.
    """
    ticker = ticker.upper()
    try:
        result = predict_next_day(ticker, task='1.1')
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f'No saved model for ticker "{ticker}" in task 1.1.'
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
    return result
