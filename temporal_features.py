from typing import List, Tuple, Optional
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
        add_elapsed (bool): Add time elapsed as the original value divided by 3000. Defaults to True.
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
            (df[field_name] % period + 1).cast(dtype).alias(feature_name)
        )
        added_features.append(feature_name)

    # Add elapsed feature (original value divided by 3000)
    if add_elapsed:
        elapsed_name = f"{prefix}Elapsed"
        df = df.with_columns(
            (df[field_name] / 3000).alias(elapsed_name)
        )
        added_features.append(elapsed_name)

    # Drop original time series column if specified
    if drop:
        df = df.drop(field_name)

    return df, added_features

def _calculate_fourier_terms(
    seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int
):
    """Calculates Fourier Terms given the seasonal cycle and max_cycle."""
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])

def add_fourier_features(
    df: pl.DataFrame,
    column_to_encode: str,
    max_value: Optional[int] = None,
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.

    Args:
        df (pl.DataFrame): The Polars dataframe which has the seasonal cycles to encode.
        column_to_encode (str): The column name containing the seasonal cycle.
        max_value (Optional[int]): The maximum value the seasonal cycle can attain, e.g., 12 for months.
            If None, it is inferred from the data, but this may not be accurate for incomplete cycles.
        n_fourier_terms (int): Number of Fourier terms to generate. Defaults to 1.
        use_32_bit (bool, optional): Whether to use float32 to reduce memory usage. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Updated DataFrame and list of added feature names.
    """
    assert column_to_encode in df.columns, f"Column '{column_to_encode}' not found in DataFrame."

    if max_value is None:
        max_value = df[column_to_encode].max().to_numpy()[0]

    # Ensure column is numeric
    seasonal_cycle = df[column_to_encode].to_numpy()
    assert np.issubdtype(seasonal_cycle.dtype, np.number), \
        f"Column '{column_to_encode}' should have numeric values."

    # Generate Fourier features
    fourier_features = _calculate_fourier_terms(seasonal_cycle, max_cycle=max_value, n_fourier_terms=n_fourier_terms)

    # Create feature names
    feature_names = [
        f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)
    ] + [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]

    # Convert to DataFrame
    fourier_df = pl.DataFrame(fourier_features, schema=feature_names)
    if use_32_bit:
        fourier_df = fourier_df.with_columns([pl.col(col).cast(pl.Float32) for col in fourier_df.columns])

    # Add Fourier features to the original DataFrame
    df = df.hstack(fourier_df)
    return df, feature_names


def bulk_add_fourier_features(
    df: pl.DataFrame,
    columns_to_encode: List[str],
    max_values: List[Optional[int]],
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pl.DataFrame, List[str]]:
    """Adds Fourier Terms for all specified seasonal cycle columns.

    Args:
        df (pl.DataFrame): Input dataframe.
        columns_to_encode (List[str]): List of column names to encode.
        max_values (List[Optional[int]]): List of maximum values for the seasonal cycles, corresponding to columns.
        n_fourier_terms (int): Number of Fourier terms to generate. Defaults to 1.
        use_32_bit (bool, optional): Whether to use float32 to reduce memory usage. Defaults to False.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Updated DataFrame and list of all added feature names.
    """
    assert len(columns_to_encode) == len(max_values), \
        "`columns_to_encode` and `max_values` must have the same length."

    added_features = []

    for column, max_value in zip(columns_to_encode, max_values):
        df, features = add_fourier_features(
            df, column_to_encode=column, max_value=max_value,
            n_fourier_terms=n_fourier_terms, use_32_bit=use_32_bit
        )
        added_features.extend(features)

    return df, added_features