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
        """
        raise NotImplementedError("Should be implemented by subclasses.")

    @classmethod
    def get_parameter_info(cls):
        """
        Returns a dictionary describing the strategy's parameters, their names, 
        default values, and default optimization ranges.
        """
        return {}

class MovingAverageCrossover(Strategy):
    """
    A strategy based on the crossover of two moving averages.
    """
    def __init__(self, short_window=40, long_window=100):
        super().__init__(name=f"MA Crossover ({short_window}/{long_window})")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")
        self.short_window = short_window
        self.long_window = long_window

    @classmethod
    def get_parameter_info(cls):
        return {
            'short_window': {'name': 'Short Window', 'default': 50, 'range': [10, 50, 5]},
            'long_window': {'name': 'Long Window', 'default': 200, 'range': [50, 200, 10]}
        }

    def generate_signals(self, data):
        price_series = data.iloc[:, 0]
        signals = pd.DataFrame(index=data.index)
        signals['price'] = price_series
        signals['short_mavg'] = price_series.rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = price_series.rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = 0.0
        signals.loc[signals.index[self.short_window:], 'signal'] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()
        signals['signal'] = signals['positions']
        signals.drop(columns=['positions', 'short_mavg', 'long_mavg'], inplace=True)
        signals.fillna(0, inplace=True)
        return signals

class RSIStrategy(Strategy):
    """
    A strategy based on the Relative Strength Index (RSI).
    """
    def __init__(self, rsi_period=14, overbought_threshold=70, oversold_threshold=30):
        super().__init__(name=f"RSI ({rsi_period}/{overbought_threshold}/{oversold_threshold})")
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    @classmethod
    def get_parameter_info(cls):
        return {
            'rsi_period': {'name': 'RSI Period', 'default': 14, 'range': [7, 21, 7]},
            'oversold_threshold': {'name': 'Oversold', 'default': 30, 'range': [20, 35, 5]},
            'overbought_threshold': {'name': 'Overbought', 'default': 70, 'range': [65, 80, 5]}
        }

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

    @classmethod
    def get_parameter_info(cls):
        return {
            'window': {'name': 'Window', 'default': 20, 'range': [10, 30, 5]},
            'std_dev': {'name': 'Std Dev', 'default': 2, 'range': [1, 3, 1]}
        }

    def generate_signals(self, data):
        price_series = data.iloc[:, 0]
        signals = pd.DataFrame(index=data.index)
        signals['price'] = price_series
        signals['middle_band'] = price_series.rolling(window=self.window).mean()
        signals['std_dev_val'] = price_series.rolling(window=self.window).std()
        signals['upper_band'] = signals['middle_band'] + (signals['std_dev_val'] * self.std_dev)
        signals['lower_band'] = signals['middle_band'] - (signals['std_dev_val'] * self.std_dev)
        signals['signal'] = 0.0
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1.0
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1.0
        signals.drop(columns=['middle_band', 'std_dev_val', 'upper_band', 'lower_band'], inplace=True)
        signals.fillna(0, inplace=True)
        return signals

class BuyAndHold(Strategy):
    def __init__(self):
        super().__init__(name="Buy and Hold")

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data.iloc[:, 0]
        signals['signal'] = 0.0
        signals['signal'].iloc[0] = 1.0
        return signals
