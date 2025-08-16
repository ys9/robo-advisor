import pandas as pd
import numpy as np

class Strategy:
    """
    A base class for trading strategies.
    """
    def __init__(self, name="Base Strategy"):
        self.name = name

    def generate_signals(self, data):
        """
        Generates trading signals for the given data.

        This method should be overridden by subclasses.

        Args:
            data (pd.DataFrame): A DataFrame with historical price data. 
                                 It is expected to have a 'Close' column.

        Returns:
            pd.DataFrame: A DataFrame with a 'signal' column containing 1 for buy, -1 for sell, and 0 for hold.
        """
        raise NotImplementedError("Should be implemented by subclasses.")

class MovingAverageCrossover(Strategy):
    """
    A strategy based on the crossover of two moving averages.
    """
    def __init__(self, short_window=40, long_window=100):
        """
        Initializes the MovingAverageCrossover strategy.

        Args:
            short_window (int): The lookback period for the short moving average.
            long_window (int): The lookback period for the long moving average.
        """
        super().__init__(name=f"Moving Average Crossover ({short_window}/{long_window})")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """
        Generates trading signals based on the moving average crossover.

        Args:
            data (pd.DataFrame): DataFrame with historical price data. 
                                 It should contain at least one column of prices.

        Returns:
            pd.DataFrame: A DataFrame with the original data, moving averages, and signals.
        """
        # Use the first column of the dataframe for calculations
        price_series = data.iloc[:, 0]

        signals = pd.DataFrame(index=data.index)
        signals['price'] = price_series
        
        # Create short and long simple moving averages
        signals['short_mavg'] = price_series.rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = price_series.rolling(window=self.long_window, min_periods=1, center=False).mean()

        # Create a signal when the two moving averages cross
        signals['signal'] = 0.0
        
        # Generate signal based on the difference between short and long mavg
        # A positive difference means the short mavg is above the long mavg
        signals.loc[signals.index[self.short_window:], 'signal'] = np.where(signals['short_mavg'][self.short_window:] 
                                                         > signals['long_mavg'][self.short_window:], 1.0, 0.0)   

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()
        
        # Rename 'positions' to 'signal' for consistency and drop the intermediate columns
        signals['signal'] = signals['positions']
        signals.drop(columns=['positions', 'short_mavg', 'long_mavg'], inplace=True)
        
        # Clean up NaN values from the start of the series
        signals.fillna(0, inplace=True)

        # Signal values: 1.0 = Buy, -1.0 = Sell, 0.0 = Hold
        return signals
