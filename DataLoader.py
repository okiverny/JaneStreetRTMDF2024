import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict, deque
from typing import List, Tuple, Dict
from Configs import MissingValueConfig
from utils import check_nulls
from autoregressive_features import (
    add_lags,
    add_rolling_features,
    add_seasonal_rolling_features,
    add_ewma
)
from temporal_features import (
    add_temporal_features,
    bulk_add_fourier_features
)

class SymbolLagsCollection:
    def __init__(self, date_buffer_size: int, missing_config: MissingValueConfig):
        self.date_buffer_size = date_buffer_size
        self.symbol_data: Dict[int, deque] = defaultdict(lambda: deque(maxlen=date_buffer_size))
        self.missing_config = missing_config

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

        #print([x for x in self.symbol_data])

    def missing_data_imputation(self, imputation_strategy: str) -> None:
        print('Implement the missing_data_imputation function')

    def construct_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Construct lagged features for the current batch of data."""
        
        # Add lags features
        #df, lag_features = add_lags(df, [968*(i+1) for i in range(2)], 'responder_6_lag_1')

        # Add rolling features
        #df, rolling_features = add_rolling_features(df, rolls=[120, 484], column='responder_6_lag_1', agg_funcs=["mean", "std"], n_shift=0, use_32_bit=True)

        # Add seasonal rolling features
        #df, season_rolling_features = add_seasonal_rolling_features(df, seasonal_periods=[968], rolls=[3], column='responder_6_lag_1', agg_funcs=["mean", "std"], n_shift=0, use_32_bit=True)

        # Add ewma features
        #df, ewma_features = add_ewma(df, 'responder_6_lag_1', spans=[242, 484, 968, 5*968], n_shift=0, use_32_bit=True)
        # df, ewma_features = add_ewma(df, 'responder_1_lag_1', spans=[5*968], n_shift=0, use_32_bit=True)

        # Add temporal features
        df, temporal_features_time = add_temporal_features(df, 'time_id', periods=[332, 725, 968], add_elapsed=True, drop=False, use_32_bit=True)
        #df, temporal_features_date = add_temporal_features(df, 'date_id', periods=[5, 20], add_elapsed=False, drop=False, use_32_bit=True)

        # Add Fourier features
        #df, fourier_features = bulk_add_fourier_features(df, columns_to_encode=['time_id_Period_968', 'date_id_Period_5', 'date_id_Period_20'], max_values=[968, 5, 20], n_fourier_terms=3, use_32_bit=True)
        df, fourier_features = bulk_add_fourier_features(df, columns_to_encode=['time_id_Period_968'], max_values=[968], n_fourier_terms=5, use_32_bit=True)
        #df.drop(['time_id_Period_968'])

        return df

    def get_lags(self, symbol_id: int) -> pl.DataFrame:
        """Retrieve historical data for a specific symbol_id."""
        combined_df = pl.concat(list(self.symbol_data[symbol_id]), how="vertical_relaxed") if symbol_id in self.symbol_data else pl.DataFrame()

        #if len(combined_df)>0:
        #    check_nulls(combined_df)

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


class RetrainData:
    def __init__(self, date_buffer_size: int):
        self.date_buffer_size = date_buffer_size
        self.retrain_data = pl.DataFrame()

    def is_full(self) -> bool:
        return len(self.retrain_data["date_id"].unique(maintain_order=True).to_list()) == self.date_buffer_size

    def add_data(self, new_data: pl.DataFrame):
        if self.retrain_data.is_empty():
            self.retrain_data = new_data
        else:
            new_date_id = new_data["date_id"].unique(maintain_order=True).to_list()[0]
            collected_dates = self.retrain_data["date_id"].unique(maintain_order=True).to_list()

            if (self.is_full()) and (new_date_id not in collected_dates):
                self.retrain_data = self.retrain_data.filter(pl.col("date_id")>collected_dates[0])
                self.retrain_data = pl.concat([self.retrain_data, new_data])
            else:
                self.retrain_data = pl.concat([self.retrain_data, new_data])

    def reset_data(self):
        self.retrain_data = pl.DataFrame()

    def __len__(self):
        """Get the number of date_id currently tracked."""
        return len(self.retrain_data["date_id"].unique(maintain_order=True).to_list())
