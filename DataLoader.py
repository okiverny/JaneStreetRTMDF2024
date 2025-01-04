import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict, deque
from typing import List, Tuple, Dict
from Configs import MissingValueConfig
from utils import check_nulls
    

class SymbolLagsCollection:
    def __init__(self, date_buffer_size: int, missing_config: MissingValueConfig, lags: List[int]):
        self.date_buffer_size = date_buffer_size
        self.symbol_data: Dict[int, deque] = defaultdict(lambda: deque(maxlen=date_buffer_size))
        self.missing_config = missing_config
        self.lags = lags

    def add_lags(self, lag_data: pl.DataFrame):
        """Add lag data to the collection for each symbol_id."""

        # Split the 'lag_data' by symbol_id and update deques
        symbol_id_given = []
        for (symbol_id,), symbol_data in lag_data.group_by("symbol_id", maintain_order=True):
            # Append the new lag data to each symbol_data collection
            self.symbol_data[symbol_id].append(symbol_data)
            symbol_id_given.append(symbol_id)

        # Fill missing data
        if self.missing_config!=None:
            for symbol_id in self.symbol_data:
                if self.missing_config.fill_symbols_daily: continue

                # duplicate the last available lags data for other symbol_id in dictionary
                if symbol_id not in symbol_id_given:
                    prev_data = self.symbol_data[symbol_id][-1]

                    # Increment date_id value
                    prev_data = prev_data.with_columns(
                        (pl.col('date_id') + 1).alias('date_id')
                    )
                    self.symbol_data[symbol_id].append(prev_data)

                # Duplicate the first available lags data to fill the collection from the left size
                while not self.is_full_symbol(symbol_id):
                    prev_data = self.symbol_data[symbol_id][0]

                    # Decrement date_id value and append left
                    prev_data = prev_data.with_columns(
                        (pl.col('date_id') - 1).alias('date_id')
                    )
                    self.symbol_data[symbol_id].appendleft(prev_data)

        print([x for x in self.symbol_data])

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
        combined_df = pl.concat(list(self.symbol_data[symbol_id]), how="vertical_relaxed") if symbol_id in self.symbol_data else pl.DataFrame()

        if len(combined_df)>0:
            check_nulls(combined_df)

        #if self.missing_config!=None:
        #    combined_df = self.missing_config.impute_missing_values(combined_df)

        return combined_df

    def is_full(self) -> bool:
        """Check if all deques are full."""
        return all(len(d) == d.maxlen for d in self.symbol_data.values())

    def is_full_symbol(self, symbol_id: int) -> bool:
        """Check if deque for symbol_id is full."""
        return len(self.symbol_data[symbol_id]) == self.symbol_data[symbol_id].maxlen

    def __len__(self):
        """Get the number of symbols currently tracked."""
        return len(self.symbol_data)