import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class Visualizer:
    """
    A class to handle the creation of financial charts using Plotly.
    """

    def plot_performance_comparison(self, performance_data, ticker):
        """
        Plots the equity curves for multiple strategies for comparison.

        Args:
            performance_data (dict): A dictionary where keys are strategy names
                                     and values are portfolio DataFrames.
            ticker (str): The ticker symbol being traded.
        """
        fig = go.Figure()

        for strategy_name, portfolio_df in performance_data.items():
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['total'],
                mode='lines',
                name=strategy_name
            ))

        fig.update_layout(
            title=f'Strategy Performance Comparison for {ticker}',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            legend_title='Strategy',
            template='plotly_dark'
        )
        fig.show()

    def plot_signals(self, signals_df, strategy_name, ticker):
        """
        Plots the price series along with buy and sell signals.

        Args:
            signals_df (pd.DataFrame): The DataFrame containing price and signal data.
            strategy_name (str): The name of the strategy being plotted.
            ticker (str): The ticker symbol being traded.
        """
        fig = make_subplots(rows=1, cols=1)

        # Plot the closing price
        fig.add_trace(go.Scatter(
            x=signals_df.index,
            y=signals_df['price'],
            mode='lines',
            name='Price'
        ), row=1, col=1)

        # Plot Buy Signals
        buy_signals = signals_df[signals_df['signal'] == 1.0]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal'
        ), row=1, col=1)

        # Plot Sell Signals
        sell_signals = signals_df[signals_df['signal'] == -1.0]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['price'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal'
        ), row=1, col=1)

        fig.update_layout(
            title=f'{strategy_name} Trading Signals for {ticker}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            legend_title='Action',
            template='plotly_dark'
        )
        fig.show()
