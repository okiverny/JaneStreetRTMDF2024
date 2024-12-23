import polars as pl
from typing import List, Tuple

ALLOWED_AGG_FUNCS = ["mean", "max", "min", "std"]

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
                df.group_by([ts_id])
                .agg([df[column].shift(lag).alias(lag_col_name).cast(dtype) if dtype else df[column].shift(lag).alias(lag_col_name)])
                .select(lag_col_name)  # Select only the lagged column
            )

    # Add new lag columns to the DataFrame
    df = df.with_columns(*new_columns)
    added_features = [col.name for col in new_columns]
    return df, added_features


def add_rolling_features(
    df: pl.DataFrame,
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add rolling statistics from the column provided and add them as new columns in the Polars DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame in which features need to be created.
        rolls (List[int]): Different windows over which the rolling aggregations to be computed.
        column (str): The column used for feature engineering.
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique ID for a time series. Defaults to None.
        n_shift (int, optional): Number of time steps to shift before computing rolling statistics.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory usage. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Returns a tuple of the new DataFrame and a list of features that were added.
    """
    assert column in df.columns, f"`{column}` should be a valid column in the provided DataFrame"
    assert all(func in ALLOWED_AGG_FUNCS for func in agg_funcs), f"`agg_funcs` should be one of {ALLOWED_AGG_FUNCS}"

    # Determine 32-bit type if needed
    dtype = pl.Float32 if use_32_bit and df.schema[column] == pl.Float64 else None

    new_columns = []
    if ts_id is None:
        # Create rolling features without grouping
        for roll in rolls:
            for agg in agg_funcs:
                col_name = f"{column}_rolling_{roll}_{agg}"
                rolling_col = (
                    df[column]
                    .shift(n_shift)
                    .rolling_mean(roll) if agg == "mean" else
                    df[column]
                    .shift(n_shift)
                    .rolling_max(roll) if agg == "max" else
                    df[column]
                    .shift(n_shift)
                    .rolling_min(roll) if agg == "min" else
                    df[column]
                    .shift(n_shift)
                    .rolling_std(roll)
                )
                new_columns.append(
                    rolling_col.alias(col_name).cast(dtype) if dtype else rolling_col.alias(col_name)
                )
    else:
        # Create rolling features with grouping
        assert ts_id in df.columns, f"`{ts_id}` should be a valid column in the provided DataFrame"
        for roll in rolls:
            for agg in agg_funcs:
                col_name = f"{column}_rolling_{roll}_{agg}"
                rolling_col = (
                    df.group_by(ts_id).agg([
                        df[column]
                        .shift(n_shift)
                        .rolling_mean(roll).alias(col_name) if agg == "mean" else
                        df[column]
                        .shift(n_shift)
                        .rolling_max(roll).alias(col_name) if agg == "max" else
                        df[column]
                        .shift(n_shift)
                        .rolling_min(roll).alias(col_name) if agg == "min" else
                        df[column]
                        .shift(n_shift)
                        .rolling_std(roll).alias(col_name)
                    ])
                )
                new_columns.append(
                    rolling_col.select(col_name).cast(dtype) if dtype else rolling_col.select(col_name)
                )

    # Add the new rolling columns to the DataFrame
    df = df.with_columns(*new_columns)
    added_features = [col.name for col in new_columns]
    return df, added_features