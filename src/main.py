import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from data_handler import DataHandler
from strategy import MovingAverageCrossover, BuyAndHold, RSIStrategy, BollingerBandsStrategy
from visualizer import Visualizer # Import the new Visualizer class

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
        pd.DataFrame: The portfolio DataFrame over time.
    """
    cash = initial_investment
    positions = 0.0
    portfolio = pd.DataFrame(index=signals.index).fillna(0.0)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = float(initial_investment)
    portfolio['total'] = float(initial_investment)

    for i in range(len(signals)):
        row_indexer = signals.index[i]
        if signals['signal'].iloc[i] == 1.0:  # Buy signal
            if cash > 0:
                positions = cash / signals['price'].iloc[i]
                cash = 0.0

        elif signals['signal'].iloc[i] == -1.0:  # Sell signal
            if positions > 0:
                cash = positions * signals['price'].iloc[i]
                positions = 0.0

        portfolio.loc[row_indexer, 'holdings'] = positions * signals['price'].iloc[i]
        portfolio.loc[row_indexer, 'cash'] = cash
        portfolio.loc[row_indexer, 'total'] = portfolio.loc[row_indexer, 'holdings'] + portfolio.loc[row_indexer, 'cash']

    # --- Performance Calculation ---
    total_return = (portfolio['total'].iloc[-1] / initial_investment) - 1
    num_years = len(signals) / 252.0
    cagr = ((portfolio['total'].iloc[-1] / initial_investment) ** (1/num_years)) - 1 if num_years > 0 else 0

    portfolio['daily_return'] = portfolio['total'].pct_change()
    annualized_volatility = portfolio['daily_return'].std() * np.sqrt(252)
    
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0

    peak = portfolio['total'].expanding(min_periods=1).max()
    drawdown = (portfolio['total']/peak) - 1
    max_drawdown = drawdown.min()

    performance_metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Final Value': portfolio['total'].iloc[-1]
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
        # --- 2. Strategy Initialization ---
        strategies = [
            BuyAndHold(),
            MovingAverageCrossover(short_window=50, long_window=200),
            RSIStrategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30),
            BollingerBandsStrategy(window=20, std_dev=2)
        ]

        results = []
        portfolio_history = {} # To store portfolio data for plotting

        # --- 3. Strategy Simulation and Comparison ---
        for strategy in strategies:
            print(f"\n--- Running Strategy: {strategy.name} ---")
            signals = strategy.generate_signals(financial_data)
            performance, portfolio = simulate_trading(signals, initial_investment)
            results.append((strategy.name, performance))
            portfolio_history[strategy.name] = portfolio

            # We'll plot the signals for one of the strategies as an example
            if isinstance(strategy, MovingAverageCrossover):
                example_signals_df = signals
                example_strategy_name = strategy.name


        # --- 4. Display Comparison Table ---
        print("\n" + "="*80)
        print(f"      Strategy Performance Comparison for {ticker_to_trade}")
        print("="*80)
        
        performance_df = pd.DataFrame([
            dict(Strategy=name, **metrics) for name, metrics in results
        ])
        performance_df.set_index('Strategy', inplace=True)

        performance_df['Final Value'] = performance_df['Final Value'].apply(lambda x: f"${x:,.2f}")
        performance_df['Total Return'] = performance_df['Total Return'].apply(lambda x: f"{x:.2%}")
        performance_df['CAGR'] = performance_df['CAGR'].apply(lambda x: f"{x:.2%}")
        performance_df['Annualized Volatility'] = performance_df['Annualized Volatility'].apply(lambda x: f"{x:.2%}")
        performance_df['Sharpe Ratio'] = performance_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        performance_df['Max Drawdown'] = performance_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")

        print(performance_df)
        print("="*80)

        # --- 5. Visualization ---
        print("\n--- Generating Visualizations ---")
        visualizer = Visualizer()
        
        # Plot 1: Performance Comparison
        visualizer.plot_performance_comparison(portfolio_history, ticker_to_trade)
        
        # Plot 2: Example of Trading Signals
        if 'example_signals_df' in locals():
            visualizer.plot_signals(example_signals_df, example_strategy_name, ticker_to_trade)


        # --- 6. Explanation of Strategy Differences ---
        print("\n--- How to Interpret the Results ---")
        print("Each strategy uses a different approach to generate buy and sell signals, leading to varied performance:")
        print("\n* Buy and Hold: A passive strategy that serves as a benchmark.")
        print("\n* Moving Average Crossover: A trend-following strategy.")
        print("\n* RSI Strategy: A momentum strategy that attempts to profit from reversals.")
        print("\n* Bollinger Bands Strategy: A volatility-based strategy assuming reversion to the mean.")

    else:
        print("Could not fetch financial data. Exiting.")
