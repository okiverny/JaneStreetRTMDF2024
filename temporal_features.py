import warnings
from typing import List, Tuple
import numpy as np
import polars as pl

def add_temporal_features(
    df: pl.DataFrame,
    field_name: str,
    periods: List[int],
    add_elapsed: bool = True,
    prefix: str = None,
    drop: bool = True,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Adds temporal features based on an integer time series column in the DataFrame.
    Args:
        df (pl.DataFrame): Input DataFrame.
        field_name (str): Name of the integer time series column.
        periods (List[int]): List of periods for creating cyclic features.
        add_elapsed (bool): Add time elapsed as the original value divided by 1000. Defaults to True.
        prefix (str): Prefix for new columns. Defaults to None.
        drop (bool): Drop the original time series column after feature creation. Defaults to True.
        use_32_bit (bool): Use 32-bit data types for memory optimization. Defaults to False.
    Returns:
        Tuple[pl.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    # Ensure field exists in the dataframe
    assert field_name in df.columns, f"Column `{field_name}` not found in DataFrame."
    
    # Set prefix for new columns
    prefix = (prefix or field_name) + "_"
    dtype = pl.Int32 if use_32_bit else pl.Int64

    added_features = []

    # Add periodic features for each period in the list
    for period in periods:
        feature_name = f"{prefix}Period_{period}"
        df = df.with_columns(
            (df[field_name] % period).cast(dtype).alias(feature_name)
        )
        added_features.append(feature_name)

    # Add elapsed feature (original value divided by 1000)
    if add_elapsed:
        elapsed_name = f"{prefix}Elapsed"
        df = df.with_columns(
            (df[field_name] / 1000).alias(elapsed_name)
        )
        added_features.append(elapsed_name)

    # Drop original time series column if specified
    if drop:
        df = df.drop(field_name)

    return df, added_features