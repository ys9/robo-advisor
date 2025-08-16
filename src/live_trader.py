import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from data_handler import DataHandler
# Import all the strategies you want to compare
from strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy

# --- Configuration ---
TICKER = 'SPY'
# Create a dictionary of strategies to choose from
STRATEGIES = {
    'Moving Average Crossover (50/200)': MovingAverageCrossover(short_window=50, long_window=200),
    'RSI (14/70/30)': RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30),
    'Bollinger Bands (20/2)': BollingerBandsStrategy(window=20, std_dev=2)
}
UPDATE_INTERVAL_SECONDS = 30 # Interval in seconds to fetch new data

# --- Initialize DataHandler and historical data ---
data_handler = DataHandler([TICKER])
end_date = datetime.now()
start_date = end_date - timedelta(days=365) # Start with one year of data

# Fetch initial historical data
historical_data = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if historical_data.empty:
    print("Could not fetch initial historical data. Exiting.")
    exit()

# Use a global dataframe to store the price history
price_history_df = historical_data.copy()

# --- Initialize Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f'Live Strategy Tracker: {TICKER}', style={'textAlign': 'center'}),
    html.Div(id='latest-price-info', style={'textAlign': 'center'}),
    
    # Add a checklist for selecting strategies
    html.Div([
        html.Label('Select Strategies to Display:'),
        dcc.Checklist(
            id='strategy-checklist',
            options=[{'label': name, 'value': name} for name in STRATEGIES.keys()],
            value=['Moving Average Crossover (50/200)'], # Default selected strategy
            inline=True,
            style={'paddingLeft': '10px'}
        )
    ], style={'textAlign': 'center', 'padding': '10px'}),

    dcc.Graph(id='live-price-chart'),
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL_SECONDS * 1000,  # in milliseconds
        n_intervals=0
    )
], style={'fontFamily': 'Arial, sans-serif'})

# --- Dash Callback for Live Updates ---
@app.callback(
    [Output('live-price-chart', 'figure'),
     Output('latest-price-info', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('strategy-checklist', 'value')] # Add the checklist as an input
)
def update_live_chart(n, selected_strategies):
    global price_history_df
    
    # --- 1. Fetch Live Price ---
    live_price = data_handler.get_live_price(TICKER)
    latest_price_text = f"Could not fetch live price."

    if live_price is not None:
        latest_price_text = f"Latest Price ({datetime.now().strftime('%H:%M:%S')}): ${live_price:.2f}"
        
        # --- 2. Append to History ---
        new_row = pd.DataFrame({TICKER: [live_price]}, index=[datetime.now()])
        price_history_df = pd.concat([price_history_df, new_row])
        price_history_df = price_history_df[~price_history_df.index.duplicated(keep='last')]

    # --- 3. Create Visualization ---
    fig = go.Figure()

    # Plot Price (only once)
    fig.add_trace(go.Scatter(
        x=price_history_df.index,
        y=price_history_df[TICKER],
        mode='lines',
        name='Price',
        line=dict(color='lightblue')
    ))

    # --- 4. Generate and Plot Signals for Selected Strategies ---
    if selected_strategies:
        for strategy_name in selected_strategies:
            strategy_obj = STRATEGIES[strategy_name]
            signals_df = strategy_obj.generate_signals(price_history_df)

            # Plot Buy Signals
            buy_signals = signals_df[signals_df['signal'] == 1.0]
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                marker=dict(color='lime', size=10, symbol='triangle-up'),
                name=f'{strategy_name} - Buy'
            ))

            # Plot Sell Signals
            sell_signals = signals_df[signals_df['signal'] == -1.0]
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name=f'{strategy_name} - Sell'
            ))

    fig.update_layout(
        title=f'Live Trading Signals for {TICKER}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        legend_title='Legend'
    )

    return fig, latest_price_text

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting live trading dashboard...")
    print(f"Open http://1227.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True)
