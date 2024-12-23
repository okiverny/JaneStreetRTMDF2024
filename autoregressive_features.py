import polars as pl
from typing import List, Tuple

def add_lags(
    df: pl.DataFrame,
    lags: List[int],
    column: str,
    ts_id: str = None,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Create lags for the specified column and add them as new columns in the Polars DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame in which features need to be created.
        lags (List[int]): List of lags to be created.
        column (str): Name of the column to be lagged.
        ts_id (str, optional): Column name of the unique ID of a time series to group by before applying the lags.
            If None, assumes the DataFrame only has a single time series. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory usage. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, List[str]]: A tuple of the new DataFrame and a list of features that were added.
    """
    assert isinstance(lags, list), "`lags` should be a list of all required lags"
    assert column in df.columns, f"`{column}` should be a valid column in the provided DataFrame"
    
    # Determine 32-bit type if needed
    if use_32_bit:
        dtype = pl.Float32 if df.schema[column] == pl.Float64 else pl.Int32
    else:
        dtype = None

    new_columns = []
    if ts_id is None:
        # Create lags without grouping
        for lag in lags:
            lag_col_name = f"{column}_lag_{lag}"
            new_columns.append(
                df[column].shift(lag).alias(lag_col_name).cast(dtype) if dtype else df[column].shift(lag).alias(lag_col_name)
            )
    else:
        # Create lags with grouping
        assert ts_id in df.columns, f"`{ts_id}` should be a valid column in the provided DataFrame"
        for lag in lags:
            lag_col_name = f"{column}_lag_{lag}"
            new_columns.append(
                df.group_by(ts_id)
                .agg([df[column].shift(lag).alias(lag_col_name).cast(dtype) if dtype else df[column].shift(lag).alias(lag_col_name)])
                .select(lag_col_name)  # Select only the lagged column
            )

    # Add new lag columns to the DataFrame
    df = df.with_columns(*new_columns)
    added_features = [col.alias for col in new_columns]
    return df, added_features

