import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataHandler:
    """
    A class to handle fetching and processing financial data from Yahoo Finance.
    """
    def __init__(self, tickers):
        """
        Initializes the DataHandler with a list of tickers.
        
        Args:
            tickers (list): A list of ETF/stock tickers.
        """
        self.tickers = tickers

    def get_historical_data(self, start_date, end_date):
        """
        Fetches historical adjusted closing prices for the handler's tickers.
        
        Args:
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame of adjusted closing prices.
        """
        print(f"Fetching historical data for {self.tickers} from {start_date} to {end_date}...")
        try:
            data = yf.download(self.tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=self.tickers[0])
            data.dropna(inplace=True)
            return data
        except Exception as e:
            print(f"An error occurred during historical data fetch: {e}")
            return pd.DataFrame()

    def get_live_price(self, ticker):
        """
        Fetches the most recent market price for a single ticker.
        
        Note: "Live" price is often the last closing price or slightly delayed
        depending on the API and market hours.
        
        Args:
            ticker (str): The single ticker to fetch the price for.

        Returns:
            float: The most recent price, or None if it fails.
        """
        if ticker not in self.tickers:
            return None
            
        print(f"Fetching live price for {ticker}...")
        try:
            # Fetch data for the last few days to ensure we get the latest price
            # 'period="5d"' is efficient for getting recent data.
            stock = yf.Ticker(ticker)
            todays_data = stock.history(period='5d')
            if not todays_data.empty:
                # .iloc[-1] gets the last available row (most recent price)
                return todays_data['Close'].iloc[-1]
            return None
        except Exception as e:
            print(f"An error occurred during live price fetch for {ticker}: {e}")
            return None

# --- Example Usage ---
if __name__ == '__main__':
    # Define the universe of tickers for our robo-advisor
    our_tickers = ['SPY', 'AGG', 'GLD', 'IJR', 'EFA']
    
    # Initialize the handler
    data_handler = DataHandler(our_tickers)
    
    # --- 1. Get Historical Data ---
    end = datetime.now()
    start = end - timedelta(days=365) # Get one year of data
    
    historical_prices = data_handler.get_historical_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if not historical_prices.empty:
        print("\n--- Successfully fetched historical data ---")
        print(historical_prices.tail()) # Show the last 5 days
        
    # --- 2. Get a "Live" Price ---
    live_price_spy = data_handler.get_live_price('SPY')
    if live_price_spy:
        print(f"\n--- Live Price Fetch ---")
        print(f"The most recent price for SPY is: ${live_price_spy:.2f}")

