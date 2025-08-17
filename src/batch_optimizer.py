import sqlite3
import json
from datetime import datetime, timedelta
import concurrent.futures
import os

from data_handler import DataHandler
from strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from optimizer import Optimizer

# --- Configuration ---
TICKER_LIST_FILE = 'tickers.txt'  # A file with one ticker symbol per line
STRATEGIES = {
    'Moving Average Crossover': MovingAverageCrossover,
    'RSI Strategy': RSIStrategy,
    'Bollinger Bands Strategy': BollingerBandsStrategy
}
OPTIMIZATION_TIME_DELTA_HOURS = 24 # Re-optimize if params are older than this
# --- CPU Configuration ---
# Set the number of CPU cores to use for parallel processing.
# Adjust this value based on your machine's specifications (e.g., 20 for your machine).
MAX_WORKERS = 10 # A safe default, can be increased to 20.

def get_tickers_from_file(filename):
    """Reads a list of tickers from a text file."""
    with open(filename, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers

def get_last_update_time(cursor, ticker, strategy_name):
    """Checks the database for the last update time of a given ticker/strategy pair."""
    cursor.execute("SELECT last_updated FROM optimization_results WHERE ticker = ? AND strategy_name = ?", (ticker, strategy_name))
    result = cursor.fetchone()
    return datetime.fromisoformat(result[0]) if result else None

def update_parameters_in_db(conn, ticker, strategy_name, params):
    """Inserts or updates the optimal parameters in the database."""
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    params_json = json.dumps(params)
    
    cursor.execute('''
        INSERT INTO optimization_results (ticker, strategy_name, parameters, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker, strategy_name) DO UPDATE SET
        parameters = excluded.parameters,
        last_updated = excluded.last_updated
    ''', (ticker, strategy_name, params_json, timestamp))
    conn.commit()

def run_optimization_for_ticker(ticker):
    """
    Performs optimization for all strategies for a single ticker.
    This function is designed to be run in its own process.
    """
    print(f"--- [Process {os.getpid()}] Processing Ticker: {ticker} ---")
    conn = sqlite3.connect('strategy_parameters.db')
    cursor = conn.cursor()
    
    data_handler = DataHandler([ticker])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    historical_data = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if historical_data.empty:
        print(f"Could not fetch data for {ticker}. Skipping.")
        conn.close()
        return f"Failed to fetch data for {ticker}"

    for name, strategy_class in STRATEGIES.items():
        last_update = get_last_update_time(cursor, ticker, name)
        
        if last_update and (datetime.now() - last_update) < timedelta(hours=OPTIMIZATION_TIME_DELTA_HOURS):
            print(f"'{name}' for {ticker} is up to date. Skipping.")
            continue
            
        print(f"Optimizing '{name}' for {ticker}...")
        param_info = strategy_class.get_parameter_info()
        param_ranges = {p: range(info['range'][0], info['range'][1] + 1, info['range'][2]) for p, info in param_info.items()}
        
        optimizer = Optimizer(strategy_class, historical_data)
        results_df = optimizer.run_optimization(param_ranges)

        if not results_df.empty:
            best_params = results_df.sort_values(by='Final Value', ascending=False).iloc[0].to_dict()
            param_names = list(param_ranges.keys())
            optimal = {p: int(best_params[p]) for p in param_names}
            update_parameters_in_db(conn, ticker, name, optimal)
            print(f"Successfully updated parameters for {ticker}/{name}: {optimal}")

    conn.close()
    return f"Successfully processed {ticker}"

if __name__ == '__main__':
    # Create a dummy tickers.txt file for the example
    # with open(TICKER_LIST_FILE, 'w') as f:
        # f.write("SPY\nAAPL\nMSFT\nGOOG\nAGG\nGLD\nTSLA\nNVDA\nJPM\nJNJ\n")
        
    tickers = get_tickers_from_file(TICKER_LIST_FILE)
    print(f"Starting batch optimization for {len(tickers)} tickers using up to {MAX_WORKERS} CPU cores.")
    
    # Use ProcessPoolExecutor to run ticker optimizations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        future_to_ticker = {executor.submit(run_optimization_for_ticker, ticker): ticker for ticker in tickers}
        
        # Process results as they are completed
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(f"Result for {ticker}: {result}")
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
        
    print("Batch optimization process complete.")
