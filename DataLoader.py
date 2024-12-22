import pandas as pd
import numpy as np
import polars as pl
from collections import deque
from typing import List, Tuple

class LagsCollection:
    def __init__(self, lags_size: int):
        self.lags_size = lags_size
        self.container = deque(maxlen=lags_size)  # A fixed-size deque for storing last N date_id's lag data
        self.combined_data = None # concatinated polars dataframe

    def __call__(self, df: pl.DataFrame, name: str) -> pl.DataFrame:
        """ Create lag features for polars dataframe {df} with name {name}."""
        for lag in self.lags:
            df = df.with_column(f'{name}_lag_{lag}', df[name].shift(lag))
        return df

    def __len__(self) -> int:
        """ Size of current collection."""
        return len(self.container)
    
    def maxlen(self):
        """ Maximum size of a deque collection."""
        return self.container.maxlen
    
    def __iadd__(self, lag_data: pl.DataFrame | pd.DataFrame):
        """ Add {lag_data} to the right side of the deque using += operator."""
        self.container.append(lag_data)
        return self

    def __combine_current_container(self):
        """ Concatinate polars dataframes. """
        self.combined_data = pl.concat(list(self.container), how='vertical_relaxed')

    def get_collection(self) -> List:
        return list(self.container)

    def get_combined_data(self) -> pl.DataFrame:
        return self.combined_data

    def is_full(self) -> bool:
        return len(self.container) == self.container.maxlen

    def update_stats(self):
        self.__combine_current_container()
    
