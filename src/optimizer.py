import pandas as pd
import itertools
from main import simulate_trading
import concurrent.futures

class Optimizer:
    """
    A class to handle the multi-threaded optimization of trading strategy parameters.
    """
    def __init__(self, strategy_class, data):
        """
        Initializes the Optimizer.

        Args:
            strategy_class: The class of the strategy to optimize.
            data (pd.DataFrame): The historical price data for backtesting.
        """
        self.strategy_class = strategy_class
        self.data = data

    def _evaluate_combination(self, combo):
        """
        Evaluates a single combination of parameters. Designed to be run in a separate thread.

        Args:
            combo (dict): A dictionary of parameter names and their values.

        Returns:
            dict: A dictionary with the results, or None if the combination is invalid.
        """
        # --- Validation logic for specific strategies ---
        if 'short_window' in combo and 'long_window' in combo:
            if combo['short_window'] >= combo['long_window']:
                return None
        
        if 'oversold_threshold' in combo and 'overbought_threshold' in combo:
            if combo['oversold_threshold'] >= combo['overbought_threshold']:
                return None

        try:
            strategy = self.strategy_class(**combo)
            signals = strategy.generate_signals(self.data)
            performance, _ = simulate_trading(signals)
            return {**combo, **performance}
        except Exception as e:
            print(f"Error evaluating {combo}: {e}")
            return None

    def run_optimization(self, parameter_ranges):
        """
        Runs a brute-force optimization using a thread pool.

        Args:
            parameter_ranges (dict): A dictionary defining the ranges for each parameter.

        Returns:
            pd.DataFrame: A DataFrame containing the results for each valid parameter combination.
        """
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]
        total_combinations = len(all_combinations)
        
        print(f"Running optimization for {self.strategy_class.__name__} with {total_combinations} combinations...")

        results = []
        # Use ThreadPoolExecutor to run evaluations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # map submits all tasks at once and returns results as they complete
            future_to_combo = {executor.submit(self._evaluate_combination, combo): combo for combo in all_combinations}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_combo)):
                print(f"Completed combination {i+1}/{total_combinations}...")
                result = future.result()
                if result:
                    results.append(result)
        
        print("Optimization complete.")
        
        if not results:
            return pd.DataFrame()
            
        final_df = pd.DataFrame(results)
        # Ensure consistent column order
        ordered_columns = param_names + [col for col in final_df.columns if col not in param_names]
        return final_df[ordered_columns]
