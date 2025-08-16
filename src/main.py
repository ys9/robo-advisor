import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from data_handler import DataHandler
from strategy import MovingAverageCrossover

# --- Phase 1: Data Acquisition ---

def get_financial_data(tickers, start_date, end_date):
    """Fetches historical adjusted closing prices for a list of stock tickers."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

# --- Phase 2: Signal-based Strategy Simulation ---

def simulate_trading(signals, initial_investment=10000):
    """
    Simulates trading based on buy/sell signals and evaluates performance.

    Args:
        signals (pd.DataFrame): DataFrame with 'price' and 'signal' columns.
                                  Signal: 1 for buy, -1 for sell, 0 for hold.
        initial_investment (float): The starting investment amount.

    Returns:
        dict: A dictionary containing key performance indicators.
    """
    cash = initial_investment
    positions = 0.0
    portfolio = pd.DataFrame(index=signals.index).fillna(0.0)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = initial_investment
    portfolio['total'] = initial_investment

    for i in range(len(signals)):
        row_indexer = signals.index[i]
        if signals['signal'].iloc[i] == 1.0:  # Buy signal
            if cash > 0:
                positions = cash / signals['price'].iloc[i]
                cash = 0.0
                print(f"{row_indexer.date()}: Buying {positions:.2f} shares at ${signals['price'].iloc[i]:.2f}")

        elif signals['signal'].iloc[i] == -1.0:  # Sell signal
            if positions > 0:
                cash = positions * signals['price'].iloc[i]
                print(f"{row_indexer.date()}: Selling {positions:.2f} shares at ${signals['price'].iloc[i]:.2f}")
                positions = 0.0

        portfolio.loc[row_indexer, 'holdings'] = positions * signals['price'].iloc[i]
        portfolio.loc[row_indexer, 'cash'] = cash
        portfolio.loc[row_indexer, 'total'] = portfolio.loc[row_indexer, 'holdings'] + portfolio.loc[row_indexer, 'cash']

    # --- Performance Calculation ---
    total_return = (portfolio['total'].iloc[-1] / initial_investment) - 1
    num_years = len(signals) / 252.0
    cagr = ((portfolio['total'].iloc[-1] / initial_investment) ** (1/num_years)) - 1

    portfolio['daily_return'] = portfolio['total'].pct_change()
    annualized_volatility = portfolio['daily_return'].std() * np.sqrt(252)
    
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0

    peak = portfolio['total'].expanding(min_periods=1).max()
    drawdown = (portfolio['total']/peak) - 1
    max_drawdown = drawdown.min()

    performance_metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio['total'].iloc[-1]
    }
    
    return performance_metrics, portfolio

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    ticker_to_trade = 'SPY'  # Example: S&P 500 ETF
    initial_investment = 10000
    end_date = datetime.now()
    start_date = datetime(end_date.year - 5, end_date.month, end_date.day)

    # --- 1. Data Fetching ---
    print(f"Fetching 5 years of historical data for: {ticker_to_trade}...")
    data_handler = DataHandler([ticker_to_trade])
    financial_data = data_handler.get_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if not financial_data.empty:
        # --- 2. Strategy and Signal Generation ---
        # You can easily swap this strategy for another one
        strategy = MovingAverageCrossover(short_window=50, long_window=200)
        print(f"\nGenerating signals using: {strategy.name}...")
        signals = strategy.generate_signals(financial_data)

        # Filter for actual trades to see the timestamps
        trade_signals = signals[signals['signal'] != 0]
        print("\n--- Generated Trade Signals (Buy/Sell Timestamps) ---")
        print(trade_signals)

        # --- 3. Simulation and Performance Evaluation ---
        print("\n--- Simulating Trading Strategy ---")
        performance, portfolio_over_time = simulate_trading(signals, initial_investment)

        # --- 4. Display Results ---
        print("\n" + "="*60)
        print(f"      Strategy Simulation Results for {ticker_to_trade}")
        print("="*60)
        print(f"Strategy: {strategy.name}\n")
        print(f"Initial Investment: ${initial_investment:,.2f}")
        print("-"*60)
        print(f"  - Final Portfolio Value: ${performance['final_value']:,.2f}")
        print(f"  - Total Return: {performance['total_return']:.2%}")
        print(f"  - Compound Annual Growth Rate (CAGR): {performance['cagr']:.2%}")
        print(f"  - Annualized Volatility (Risk): {performance['annualized_volatility']:.2%}")
        print(f"  - Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"  - Maximum Drawdown: {performance['max_drawdown']:.2%}")
        print("="*60)
    else:
        print("Could not fetch financial data. Exiting.")

