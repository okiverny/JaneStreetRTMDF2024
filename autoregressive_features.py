import polars as pl
from typing import List, Tuple, Callable, Dict

ALLOWED_AGG_FUNCS = ["mean", "max", "min", "std"]

SEASONAL_ROLLING_MAP = {
    "mean": lambda s: s.mean(),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
    "std": lambda s: s.std(),
}

def seasonal_rolling(
    series: pl.Series,
    season_length: int,
    window_size: int,
    agg_func: Callable
) -> pl.Series:
    """
    Apply a seasonal rolling operation to a Polars Series.

    Args:
        series (pl.Series): Input series to operate on.
        season_length (int): Number of steps in a seasonal period.
        window_size (int): Rolling window size.
        agg_func (Callable): Aggregation function to apply.

    Returns:
        pl.Series: Resulting series after applying the seasonal rolling operation.
    """
    n = len(series)
    result = [None] * n
    for i in range(window_size * season_length, n):
        seasonal_window = series[i - (window_size * season_length):i:season_length]
        result[i] = agg_func(seasonal_window)
    return pl.Series(result)


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
    dtype = pl.Float32 if use_32_bit else None

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

def add_seasonal_rolling_features(
    df: pl.DataFrame,
    seasonal_periods: List[int],
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add seasonal rolling statistics from the column provided and add them as new columns in the Polars DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        seasonal_periods (List[int]): List of seasonal periods.
        rolls (List[int]): List of rolling window sizes.
        column (str): Column for which features are created.
        agg_funcs (List[str], optional): Aggregation functions to apply. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique ID for a time series. Defaults to None.
        n_shift (int, optional): Number of seasonal shifts to apply before rolling. Defaults to 1.
        use_32_bit (bool, optional): Reduce memory by using 32-bit types. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Updated DataFrame and list of new column names.
    """
    assert column in df.columns, f"`{column}` must be a valid column in the DataFrame."
    assert all(agg in SEASONAL_ROLLING_MAP for agg in agg_funcs), \
        f"`agg_funcs` must be one of {list(SEASONAL_ROLLING_MAP.keys())}"

    dtype = pl.Float32 if use_32_bit else None
    new_columns = []

    for sp in seasonal_periods:
        for roll in rolls:
            for agg in agg_funcs:
                col_name = f"{column}_{sp}_seasonal_rolling_{roll}_{agg}"
                if ts_id is None:
                    # Single time series
                    shifted = df[column].shift(n_shift * sp)
                    seasonal_rolled = seasonal_rolling(
                        shifted, season_length=sp, window_size=roll, agg_func=SEASONAL_ROLLING_MAP[agg]
                    )
                    seasonal_rolled = seasonal_rolled.cast(dtype) if dtype else seasonal_rolled
                    df = df.with_columns(seasonal_rolled.alias(col_name))
                else:
                    # Grouped by time series ID
                    df = df.with_columns(
                        df.groupby(ts_id).agg([
                            seasonal_rolling(
                                df[column].shift(n_shift * sp),
                                season_length=sp,
                                window_size=roll,
                                agg_func=SEASONAL_ROLLING_MAP[agg]
                            ).alias(col_name)
                        ])
                    )
                new_columns.append(col_name)

    return df, new_columns


def add_ewma(
    df: pl.DataFrame,
    column: str,
    alphas: List[float] = None,
    spans: List[float] = None,
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add Exponentially Weighted Moving Averages (EWMA) as new features in the Polars DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column (str): Column to compute EWMA on.
        alphas (List[float]): List of alpha values (smoothing parameters).
        spans (List[float]): List of spans for EWMA. Span is converted to alpha internally.
        ts_id (str): Time series ID column to group by before applying EWMA.
        n_shift (int): Number of shifts to apply to avoid data leakage.
        use_32_bit (bool): Flag to use float32 for memory optimization.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Updated DataFrame and list of new column names.
    """
    assert column in df.columns, f"`{column}` must be a valid column in the DataFrame."

    # Convert spans to alphas if spans are provided
    if spans is not None:
        assert isinstance(spans, list), "`spans` must be a list of period spans."
        alphas = [2 / (1 + span) for span in spans]

    if not alphas:
        raise ValueError("Either `alphas` or `spans` must be provided.")

    dtype = pl.Float32 if use_32_bit else pl.Float64
    new_columns = []

    for alpha in alphas:
        col_name = f"{column}_ewma_alpha_{alpha:.4f}"

        if ts_id is None:
            # Single time series
            ewma_series = (
                df[column]
                .shift(n_shift)
                .ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
                .cast(dtype)
            )
            df = df.with_columns(ewma_series.alias(col_name))
        else:
            # Group by ts_id for multiple time series
            df = df.with_columns(
                df.groupby(ts_id)
                .agg(
                    pl.col(column)
                    .shift(n_shift)
                    .ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
                    .alias(col_name)
                )
                .explode(col_name)
            )

        new_columns.append(col_name)

    return df, new_columns