import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import sqlite3

from data_handler import DataHandler
from strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy

# --- Configuration ---
STRATEGIES = {
    'Moving Average Crossover': MovingAverageCrossover,
    'RSI Strategy': RSIStrategy,
    'Bollinger Bands Strategy': BollingerBandsStrategy
}
STRATEGY_VISUALS = {
    'Moving Average Crossover': {'color': 'cyan', 'symbol': 'circle'},
    'RSI Strategy': {'color': 'magenta', 'symbol': 'diamond'},
    'Bollinger Bands Strategy': {'color': 'yellow', 'symbol': 'square'}
}
DB_FILE = 'strategy_parameters.db'
PRICE_UPDATE_INTERVAL_SECONDS = 30

# --- Global State ---
app_state = {
    'price_history': pd.DataFrame()
}

# --- Helper Functions ---
def get_params_from_db(ticker, strategy_name):
    """Fetches the latest optimal parameters from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT parameters, last_updated FROM optimization_results WHERE ticker = ? AND strategy_name = ?", (ticker, strategy_name))
    result = cursor.fetchone()
    conn.close()
    if result:
        return json.loads(result[0]), datetime.fromisoformat(result[1])
    return None, None

# --- Initialize Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Live Strategy Tracker (with Pre-Optimized Parameters)', style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label('Ticker Symbol:'),
            dcc.Input(id='ticker-input', value='SPY', type='text', style={'width': '100%'})
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Strategies to Display:'),
            dcc.Checklist(
                id='strategy-checklist',
                options=[{'label': name, 'value': name} for name in STRATEGIES.keys()],
                value=['Moving Average Crossover'],
                inline=True
            ),
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'width': '80%', 'margin': '0 auto'}),

    html.Div(id='status-bar', style={'textAlign': 'center', 'padding': '10px'}),
    dcc.Graph(id='live-price-chart'),
    
    dcc.Interval(id='price-interval', interval=PRICE_UPDATE_INTERVAL_SECONDS * 1000, n_intervals=0)
])

# --- Main Callback for Live Updates ---
@app.callback(
    [Output('live-price-chart', 'figure'),
     Output('status-bar', 'children')],
    [Input('price-interval', 'n_intervals'),
     Input('ticker-input', 'value'),
     Input('strategy-checklist', 'value')]
)
def update_live_chart(n, ticker, selected_strategies):
    if not ticker:
        return go.Figure(), "Please enter a ticker symbol."

    ticker = ticker.upper()
    data_handler = DataHandler([ticker])

    if app_state['price_history'].empty or ticker not in app_state['price_history'].columns:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        app_state['price_history'] = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    live_price = data_handler.get_live_price(ticker)
    if live_price is not None:
        new_row = pd.DataFrame({ticker: [live_price]}, index=[datetime.now()])
        app_state['price_history'] = pd.concat([app_state['price_history'], new_row])
        app_state['price_history'] = app_state['price_history'][~app_state['price_history'].index.duplicated(keep='last')]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=app_state['price_history'].index,
        y=app_state['price_history'][ticker],
        mode='lines', name='Price', line=dict(color='lightblue')
    ))

    status_texts = [f"Last Price: ${live_price:.2f}" if live_price else "Fetching price..."]
    
    if selected_strategies:
        for name in selected_strategies:
            params, last_updated = get_params_from_db(ticker, name)
            param_status = f"(Optimal as of {last_updated.strftime('%Y-%m-%d %H:%M')})" if params else "(Default)"
            
            if not params:
                param_info = STRATEGIES[name].get_parameter_info()
                params = {p: info['default'] for p, info in param_info.items()}

            strategy_obj = STRATEGIES[name](**params)
            signals_df = strategy_obj.generate_signals(app_state['price_history'])
            visuals = STRATEGY_VISUALS.get(name, {'color': 'white', 'symbol': 'circle'})
            
            buy_signals = signals_df[signals_df['signal'] == 1.0]
            sell_signals = signals_df[signals_df['signal'] == -1.0]
            
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], mode='markers', marker=dict(color=visuals['color'], size=10, symbol=f"{visuals['symbol']}-open-dot"), name=f'{name} - Buy'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], mode='markers', marker=dict(color=visuals['color'], size=10, symbol=visuals['symbol']), name=f'{name} - Sell'))
            
            status_texts.append(f"{name} {param_status}: {params}")

    fig.update_layout(title=f'Live Signals for {ticker}', template='plotly_dark', legend_title="Legend")
    
    status_bar_text = f" | ".join(status_texts)
    return fig, status_bar_text

# --- Main Execution ---
if __name__ == '__main__':
    app.run_server(debug=False)
