import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import threading
from io import StringIO

from data_handler import DataHandler
from strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from optimizer import Optimizer

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
PARAMS_FILE = 'optimal_params.json'
PRICE_UPDATE_INTERVAL_SECONDS = 30
OPTIMIZATION_INTERVAL_MINUTES = 5

# --- Global State Management ---
app_state = {
    'price_history': pd.DataFrame(),
    'optimal_params': {},
    'optimization_threads': {}, # Manages multiple threads
    'last_optimization_time': 'Never'
}

# --- Helper Functions ---
def load_optimal_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_optimal_params(params):
    # Use a lock to prevent race conditions when saving the file from multiple threads
    lock = threading.Lock()
    with lock:
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=4)

def optimize_single_strategy(ticker, strategy_name, strategy_class):
    """Optimizes a single strategy and updates the shared JSON file."""
    print(f"[{datetime.now()}] Starting optimization for {strategy_name} on {ticker}...")
    
    data_handler = DataHandler([ticker])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    historical_data = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if historical_data.empty:
        print(f"Could not fetch data for {ticker}. Aborting optimization for {strategy_name}.")
        return

    param_info = strategy_class.get_parameter_info()
    param_ranges = {p: range(info['range'][0], info['range'][1] + 1, info['range'][2]) for p, info in param_info.items()}
    
    optimizer = Optimizer(strategy_class, historical_data)
    results_df = optimizer.run_optimization(param_ranges)

    if not results_df.empty:
        best_params = results_df.sort_values(by='Final Value', ascending=False).iloc[0].to_dict()
        param_names = list(param_ranges.keys())
        optimal = {p: int(best_params[p]) for p in param_names}
        
        all_saved_params = load_optimal_params()
        key = f"{ticker.upper()}_{strategy_name}"
        all_saved_params[key] = optimal
        save_optimal_params(all_saved_params)
        
    app_state['last_optimization_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{datetime.now()}] Optimization for {strategy_name} on {ticker} complete.")


# --- Initialize Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Live Adaptive Strategy Tracker', style={'textAlign': 'center'}),
    
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
    
    dcc.Interval(id='price-interval', interval=PRICE_UPDATE_INTERVAL_SECONDS * 1000, n_intervals=0),
    dcc.Interval(id='optimization-interval', interval=OPTIMIZATION_INTERVAL_MINUTES * 60 * 1000, n_intervals=0)
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
    
    app_state['optimal_params'] = load_optimal_params()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=app_state['price_history'].index,
        y=app_state['price_history'][ticker],
        mode='lines', name='Price', line=dict(color='lightblue')
    ))

    status_texts = [f"Last Price: ${live_price:.2f}" if live_price else "Fetching price..."]
    
    if selected_strategies:
        for name in selected_strategies:
            key = f"{ticker}_{name}"
            params = app_state['optimal_params'].get(key)
            param_status = "(Optimal)"
            if not params:
                param_info = STRATEGIES[name].get_parameter_info()
                params = {p: info['default'] for p, info in param_info.items()}
                param_status = "(Default)"

            strategy_obj = STRATEGIES[name](**params)
            signals_df = strategy_obj.generate_signals(app_state['price_history'])
            visuals = STRATEGY_VISUALS.get(name, {'color': 'white', 'symbol': 'circle'})
            
            buy_signals = signals_df[signals_df['signal'] == 1.0]
            sell_signals = signals_df[signals_df['signal'] == -1.0]
            
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], mode='markers', marker=dict(color=visuals['color'], size=10, symbol=f"{visuals['symbol']}-open-dot"), name=f'{name} - Buy'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], mode='markers', marker=dict(color=visuals['color'], size=10, symbol=visuals['symbol']), name=f'{name} - Sell'))
            
            status_texts.append(f"{name} {param_status}: {params}")

    fig.update_layout(title=f'Live Adaptive Signals for {ticker}', template='plotly_dark', legend_title="Legend")
    
    # Check status of all optimization threads
    running_optimizations = [name for name, thread in app_state['optimization_threads'].items() if thread.is_alive()]
    opt_status = f"RUNNING ({', '.join(running_optimizations)})" if running_optimizations else "IDLE"
    
    status_bar_text = f"Live Price: {status_texts[0]} | Optimization: {opt_status} (Last Run: {app_state['last_optimization_time']}) | {' | '.join(status_texts[1:])}"
    return fig, status_bar_text

# --- Callback for Periodic Optimization ---
@app.callback(
    Output('ticker-input', 'style'), # Dummy output
    [Input('optimization-interval', 'n_intervals'),
     Input('ticker-input', 'value')],
    prevent_initial_call=True
)
def trigger_optimization(n, ticker):
    if not ticker:
        return {'width': '100%'}

    # Launch a new thread for each strategy if it's not already running
    for name, strategy_class in STRATEGIES.items():
        thread = app_state['optimization_threads'].get(name)
        if not thread or not thread.is_alive():
            new_thread = threading.Thread(target=optimize_single_strategy, args=(ticker.upper(), name, strategy_class))
            app_state['optimization_threads'][name] = new_thread
            new_thread.start()
        else:
            print(f"Optimization for {name} is already in progress. Skipping.")
            
    return {'width': '100%'}

# --- Main Execution ---
if __name__ == '__main__':
    # On startup, launch an optimization thread for each strategy
    for name, strategy_class in STRATEGIES.items():
        thread = threading.Thread(target=optimize_single_strategy, args=('SPY', name, strategy_class))
        app_state['optimization_threads'][name] = thread
        thread.start()
    
    app.run_server(debug=False)
