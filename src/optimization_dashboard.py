import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from io import StringIO # Import StringIO

from data_handler import DataHandler
from strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from optimizer import Optimizer

# --- Configuration ---
STRATEGIES = {
    'Moving Average Crossover': MovingAverageCrossover,
    'RSI Strategy': RSIStrategy,
    'Bollinger Bands Strategy': BollingerBandsStrategy
}
PARAMS_FILE = 'optimal_params.json'

# --- Helper Functions ---
def load_optimal_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_optimal_params(params):
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=4)

# --- Initialize Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id='optimization-results-store'),
    html.H1('Trading Strategy Parameter Optimizer', style={'textAlign': 'center'}),
    
    # --- Control Panel ---
    html.Div([
        html.Div([
            html.Label('Ticker Symbol:'),
            dcc.Input(id='ticker-input', value='SPY', type='text', style={'width': '100%'}),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label('Select Strategy:'),
            dcc.Dropdown(
                id='strategy-selector',
                options=[{'label': name, 'value': name} for name in STRATEGIES.keys()],
                value='Moving Average Crossover'
            ),
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    html.Div(id='saved-params-display', style={'textAlign': 'center', 'padding': '10px', 'fontStyle': 'italic'}),
    html.Div(id='parameter-inputs-container', style={'textAlign': 'center', 'padding': '10px'}),
    
    html.Div(
        html.Button('Run Optimization', id='run-opt-button', n_clicks=0, style={'marginTop': '10px'}),
        style={'textAlign': 'center'}
    ),
    
    # --- Results Display ---
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=[
            html.Div(id='optimization-results-table'),
            dcc.Graph(id='optimization-heatmap')
        ]
    )
])

# --- Callback to generate dynamic parameter inputs ---
@app.callback(
    Output('parameter-inputs-container', 'children'),
    Input('strategy-selector', 'value')
)
def render_parameter_inputs(strategy_name):
    strategy_class = STRATEGIES[strategy_name]
    param_info = strategy_class.get_parameter_info()
    
    inputs = [html.H4('Set Parameter Ranges for Optimization')]
    for param, info in param_info.items():
        inputs.extend([
            html.Label(f"{info['name']}:  Start, End, Step"), html.Br(),
            dcc.Input(id={'type': 'param-start', 'index': param}, type='number', value=info['range'][0], style={'marginRight': '5px'}),
            dcc.Input(id={'type': 'param-end', 'index': param}, type='number', value=info['range'][1], style={'marginRight': '5px'}),
            dcc.Input(id={'type': 'param-step', 'index': param}, type='number', value=info['range'][2]),
            html.Br(), html.Br()
        ])
    return html.Div(inputs)

# --- Callback to display saved parameters ---
@app.callback(
    Output('saved-params-display', 'children'),
    [Input('ticker-input', 'value'), Input('strategy-selector', 'value')]
)
def display_saved_params(ticker, strategy_name):
    if not ticker: return ""
    params = load_optimal_params()
    key = f"{ticker.upper()}_{strategy_name}"
    if key in params:
        return f"Saved Optimal Params for {key}: {json.dumps(params[key])}"
    return f"No saved parameters found for {key}."

# --- Main callback to run optimization ---
@app.callback(
    Output('optimization-results-store', 'data'),
    Input('run-opt-button', 'n_clicks'),
    [State('ticker-input', 'value'),
     State('strategy-selector', 'value'),
     State({'type': 'param-start', 'index': ALL}, 'value'),
     State({'type': 'param-end', 'index': ALL}, 'value'),
     State({'type': 'param-step', 'index': ALL}, 'value'),
     State({'type': 'param-start', 'index': ALL}, 'id')]
)
def run_optimization_callback(n_clicks, ticker, strategy_name, starts, ends, steps, ids):
    if n_clicks == 0 or not ticker:
        return {}

    # Fetch data for the selected ticker
    data_handler = DataHandler([ticker.upper()])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    historical_data = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if historical_data.empty:
        return {'error': f'Could not fetch data for {ticker}.'}

    strategy_class = STRATEGIES[strategy_name]
    param_ranges = {p_id['index']: range(starts[i], ends[i] + 1, steps[i]) for i, p_id in enumerate(ids)}

    # Initialize the optimizer with only the strategy and data
    optimizer = Optimizer(strategy_class, historical_data)
    # Pass the parameter ranges to the run_optimization method
    results_df = optimizer.run_optimization(param_ranges)

    if results_df.empty:
        return {'error': 'No valid results from optimization.'}

    # Save the best parameters
    best_params = results_df.sort_values(by='Final Value', ascending=False).iloc[0].to_dict()
    param_names = list(param_ranges.keys())
    optimal = {p: int(best_params[p]) for p in param_names} # Convert numpy types to native int for JSON
    
    all_saved_params = load_optimal_params()
    key = f"{ticker.upper()}_{strategy_name}"
    all_saved_params[key] = optimal
    save_optimal_params(all_saved_params)

    return {'results': results_df.to_json(), 'params': param_names, 'error': None}

# --- Callback to update UI with results ---
@app.callback(
    [Output('optimization-results-table', 'children'),
     Output('optimization-heatmap', 'figure')],
    Input('optimization-results-store', 'data')
)
def update_results_ui(data):
    if not data:
        return html.Div("Enter a ticker, select a strategy, set ranges, and click 'Run Optimization'."), go.Figure()

    if data.get('error'):
        return html.Div(f"Error: {data['error']}"), go.Figure()

    # --- FIX IS HERE ---
    # Wrap the JSON string in StringIO to avoid the FutureWarning
    results_df = pd.read_json(StringIO(data['results']))
    param_names = data['params']

    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in results_df.columns],
        data=results_df.to_dict('records'),
        sort_action="native", page_size=10,
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    )

    fig = go.Figure()
    if len(param_names) == 2:
        try:
            heatmap_data = results_df.pivot(index=param_names[1], columns=param_names[0], values='Final Value')
            fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
            fig.update_layout(title='Optimization Heatmap (Final Portfolio Value)', xaxis_title=param_names[0], yaxis_title=param_names[1], template='plotly_dark')
        except Exception as e:
            fig.update_layout(title=f'Could not generate heatmap: {e}', template='plotly_dark')
    else:
        fig.update_layout(title='Heatmap is only available for 2-parameter optimizations.', template='plotly_dark')

    return html.Div([html.H3("Optimization Results"), table]), fig

# --- Main Execution ---
if __name__ == '__main__':
    app.run_server(debug=True)
