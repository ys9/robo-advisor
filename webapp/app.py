from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# --- Data Fetching Function ---
def get_historical_data_for_web(ticker):
    """Fetches intraday (1-minute) historical data for the last 5 days."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="1m")
        if data.empty:
            return pd.DataFrame(), f"No intraday data found for ticker '{ticker}'."
        data.dropna(inplace=True)
        return data, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), f"An error occurred while fetching data for {ticker}."

# --- Algorithmic Trading Strategy ---
def generate_sma_signals(data, short_window=20, long_window=60):
    """Generates trading signals based on a Simple Moving Average (SMA) crossover strategy."""
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Initialize the Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # --- MODIFIED: Handle form inputs for strategy parameters ---
    default_ticker = 'SPY'
    ticker = request.form.get('ticker', default_ticker)
    
    # Check if the strategy should be enabled
    strategy_enabled = request.form.get('strategy_enabled') == 'on'
    
    # Get window sizes from form, with defaults
    try:
        short_window = int(request.form.get('short_window', 20))
        long_window = int(request.form.get('long_window', 60))
    except (ValueError, TypeError):
        short_window = 20
        long_window = 60
    
    df, error_message = get_historical_data_for_web(ticker)
    
    fig = go.Figure()

    if not df.empty:
        # Add the main price line first
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='skyblue', width=2)))
        
        chart_title = f'{ticker} Intraday Price History'

        # --- MODIFIED: Conditionally add strategy plot elements ---
        if strategy_enabled:
            signals = generate_sma_signals(df, short_window, long_window)

            fig.add_trace(go.Scatter(x=signals.index, y=signals['short_mavg'], mode='lines', name=f'{short_window}-Min SMA', line=dict(color='orange', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=signals.index, y=signals['long_mavg'], mode='lines', name=f'{long_window}-Min SMA', line=dict(color='fuchsia', width=1.5, dash='dot')))

            buy_signals = signals[signals['positions'] == 1.0]
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=signals['short_mavg'][buy_signals.index],
                mode='markers', name='Buy Signal',
                marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(width=1, color='black'))
            ))

            sell_signals = signals[signals['positions'] == -1.0]
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=signals['short_mavg'][sell_signals.index],
                mode='markers', name='Sell Signal',
                marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='black'))
            ))
            chart_title = f'{ticker} with SMA Crossover ({short_window}/{long_window})'
        
        fig.update_layout(
            title=chart_title,
            xaxis_title='Date and Time',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            legend_title_text='Legend'
        )
    else:
        fig.update_layout(title=f'Data not available for {ticker}', template='plotly_dark')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    available_tickers = ['SPY', 'AGG', 'GLD', 'IJR', 'EFA', 'AAPL', 'TSLA', 'GOOGL']

    return render_template('index.html', graphJSON=graphJSON, tickers=available_tickers, selected_ticker=ticker,
                           error_message=error_message, strategy_enabled=strategy_enabled,
                           short_window=short_window, long_window=long_window)


# --- API Endpoint for Live Data ---
@app.route('/live-update/<string:ticker>')
def live_update(ticker):
    """Provides the most recent price data point for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='5d', interval='1m')
        if data.empty:
            return jsonify({'error': 'No data found'}), 404
        
        latest = data.iloc[-1]
        
        update = {
            'timestamp': latest.name.isoformat(),
            'price': latest['Close']
        }
        return jsonify(update)
    except Exception as e:
        print(f"Live update error for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
