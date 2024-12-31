import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict, deque
from typing import List, Tuple, Dict
    

class SymbolLagsCollection:
    def __init__(self, date_buffer_size: int, lags: List[int]):
        self.date_buffer_size = date_buffer_size
        self.symbol_data: Dict[int, deque] = defaultdict(lambda: deque(maxlen=date_buffer_size))
        self.lags = lags

    def add_lags(self, lag_data: pl.DataFrame):
        """Add lag data to the collection for each symbol_id."""
        # Split the data by symbol_id and update deques
        for (symbol_id,), symbol_data in lag_data.group_by("symbol_id", maintain_order=True):
            self.symbol_data[symbol_id].append(symbol_data)

    def missing_data_imputation(self, imputation_strategy: str) -> None:
        print('Implement the missing_data_imputation function')

    def construct_features(self, current_data: pl.DataFrame, lags: List[int], feature_cols: List[str]) -> pl.DataFrame:
        """Construct lagged features for the current batch of data."""
        lag_features = []
        for (symbol_id,), symbol_df in current_data.group_by("symbol_id", maintain_order=True):
            # Combine historical data for the symbol
            combined_symbol_data = pl.concat(list(self.symbol_data[symbol_id]), how="vertical_relaxed") if symbol_id in self.symbol_data else pl.DataFrame()

            # Add lagged features for the symbol
            for col in feature_cols:
                for lag in lags:
                    if not combined_symbol_data.is_empty():
                        current_data = current_data.with_columns(
                            current_data[col].shift(lag).alias(f"{col}_lag_{lag}")
                        )

            lag_features.append(current_data)

        return pl.concat(lag_features, how="vertical_relaxed")

    def get_lags(self, symbol_id: int) -> pl.DataFrame:
        """Retrieve historical data for a specific symbol_id."""
        return pl.concat(list(self.symbol_data[symbol_id]), how="vertical_relaxed") if symbol_id in self.symbol_data else pl.DataFrame()

    def is_full(self) -> bool:
        """Check if all deques are full."""
        return all(len(d) == d.maxlen for d in self.symbol_data.values())

    def is_full_symbol(self, symbol_id: int) -> bool:
        """Check if deque for symbol_id is full."""
        return len(self.symbol_data[symbol_id]) == self.symbol_data[symbol_id].maxlen

    def __len__(self):
        """Get the number of symbols currently tracked."""
        return len(self.symbol_data)