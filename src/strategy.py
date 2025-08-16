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


class BuyAndHold(Strategy):
    """
    A simple buy and hold strategy. Buys on the first day and holds.
    """
    def __init__(self):
        super().__init__(name="Buy and Hold")

    def generate_signals(self, data):
        """
        Generates a single buy signal on the first day.

        Args:
            data (pd.DataFrame): DataFrame with historical price data.

        Returns:
            pd.DataFrame: A DataFrame with a 'signal' column.
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data.iloc[:, 0]
        signals['signal'] = 0.0
        signals['signal'].iloc[0] = 1.0  # Buy on the first day
        return signals

class RSIStrategy(Strategy):
    """
    A strategy based on the Relative Strength Index (RSI).
    """
    def __init__(self, rsi_period=14, overbought_threshold=70, oversold_threshold=30):
        super().__init__(name=f"RSI Strategy ({rsi_period}/{overbought_threshold}/{oversold_threshold})")
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def generate_signals(self, data):
        price_series = data.iloc[:, 0]
        signals = pd.DataFrame(index=data.index)
        signals['price'] = price_series

        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals['rsi'] = rsi
        signals['signal'] = 0.0
        signals.loc[signals['rsi'] < self.oversold_threshold, 'signal'] = 1.0
        signals.loc[signals['rsi'] > self.overbought_threshold, 'signal'] = -1.0
        
        signals.drop(columns=['rsi'], inplace=True)
        signals.fillna(0, inplace=True)
        return signals

class BollingerBandsStrategy(Strategy):
    """
    A strategy based on Bollinger Bands.
    """
    def __init__(self, window=20, std_dev=2):
        super().__init__(name=f"Bollinger Bands ({window}/{std_dev})")
        self.window = window
        self.std_dev = std_dev

    def generate_signals(self, data):
        price_series = data.iloc[:, 0]
        signals = pd.DataFrame(index=data.index)
        signals['price'] = price_series

        signals['middle_band'] = price_series.rolling(window=self.window).mean()
        signals['std_dev'] = price_series.rolling(window=self.window).std()
        signals['upper_band'] = signals['middle_band'] + (signals['std_dev'] * self.std_dev)
        signals['lower_band'] = signals['middle_band'] - (signals['std_dev'] * self.std_dev)

        signals['signal'] = 0.0
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1.0
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1.0
        
        signals.drop(columns=['middle_band', 'std_dev', 'upper_band', 'lower_band'], inplace=True)
        signals.fillna(0, inplace=True)
        return signals