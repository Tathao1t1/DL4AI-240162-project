"""
streamlit_app.py — Streamlit UI for DL4AI Task 1.1 stock price prediction.

Run:
    cd /Users/ttt/Downloads/DL4AI-240162-project
    .venv/bin/streamlit run app/streamlit_app.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from app.predictor import predict_next_day, list_available_tickers, fetch_recent, add_technical_indicators

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'DL4AI Stock Predictor',
    page_icon  = '📈',
    layout     = 'wide',
)

st.title('📈 DL4AI — Task 1.1: Next-Day Stock Price Prediction')
st.caption('LSTM model trained on 19 technical indicators. Predictions fetched live from yfinance.')

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header('Settings')

available = list_available_tickers('1.1')
ticker    = st.sidebar.selectbox('Select Ticker', available, index=available.index('AAPL') if 'AAPL' in available else 0)
n_history = st.sidebar.slider('Price history to display (days)', 60, 365, 180)

predict_btn = st.sidebar.button('🔮 Predict Next Day', type='primary', use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader('Model Prediction')
    result_placeholder = st.empty()

with col1:
    st.subheader(f'{ticker} — Recent Price History')
    chart_placeholder = st.empty()

# ── Load and display recent price chart ───────────────────────────────────────
@st.cache_data(ttl=300)
def get_history(ticker, days):
    df = fetch_recent(ticker, window=days + 60)
    df = add_technical_indicators(df)
    return df.tail(days).reset_index(drop=True)

df_hist = get_history(ticker, n_history)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_hist['Date'], y=df_hist['Close'],
    mode='lines', name='Close Price',
    line=dict(color='steelblue', width=2),
))
fig.update_layout(
    xaxis_title='Date', yaxis_title='Price (USD)',
    height=380, margin=dict(l=0, r=0, t=20, b=0),
    hovermode='x unified',
)
chart_placeholder.plotly_chart(fig, use_container_width=True)

# ── Run prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(f'Running LSTM model for {ticker} …'):
        try:
            result = predict_next_day(ticker, task='1.1')

            direction_emoji = '🟢' if result['direction'] == 'UP' else '🔴'
            change          = result['predicted_price'] - result['last_close']
            change_pct      = result['predicted_return_pct']

            with result_placeholder.container():
                st.metric(
                    label    = f'Predicted Close  ({direction_emoji} {result["direction"]})',
                    value    = f'${result["predicted_price"]:.2f}',
                    delta    = f'{change:+.2f} ({change_pct:+.3f}%)',
                )
                st.metric('Last Close',      f'${result["last_close"]:.2f}')
                st.metric('90% CI Low',      f'${result["price_low_90ci"]:.2f}')
                st.metric('90% CI High',     f'${result["price_high_90ci"]:.2f}')
                st.caption(f'As of: {result["as_of_date"]}')

            # Add prediction point to chart
            fig.add_trace(go.Scatter(
                x    = [result['as_of_date']],
                y    = [result['predicted_price']],
                mode = 'markers',
                name = 'Prediction',
                marker = dict(color='tomato', size=12, symbol='star'),
            ))
            fig.add_hrect(
                y0=result['price_low_90ci'], y1=result['price_high_90ci'],
                fillcolor='tomato', opacity=0.1,
                annotation_text='90% CI', annotation_position='top left',
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Raw JSON expander
            with st.expander('Raw API response'):
                st.json(result)

        except Exception as e:
            st.error(f'Prediction failed: {e}')

else:
    with result_placeholder.container():
        st.info('Click **Predict Next Day** to run the model.')

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption('CS313 DL4AI Final Project · Task 1.1 · LSTM Next-Day Close Price Prediction')
