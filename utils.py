import polars as pl
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def R2w_metric(y_true, y_pred, weight):
    """
    R^2 metric with weights from sklearn
    """
    is_higher_better = True
    return 'R2w_score', r2_score(y_true, y_pred, sample_weight = weight), is_higher_better

def R2_metric(y_true, y_pred):
    """
    R^2 metric from sklearn
    """
    is_higher_better = True
    return 'r2_score', r2_score(y_true, y_pred), is_higher_better

def calculate_metrics(
    y_true: pl.Series, y_pred: pl.Series, name: str, weights: pl.Series = None
):
    """Method to calculate the metrics given the actual and predicted series

    Args:
        y (pl.Series): Actual target with datetime index
        y_pred (pl.Series): Predictions with datetime index
        name (str): Name or identification for the model
        weights (pl.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.

    Returns:
        Dict: Dictionary with MAE, MSE, and R2
    """
    return {
        "Algorithm": name,
        "R2": r2_score(y_true, y_pred, sample_weight = weights),
        "MAE": mean_absolute_error(y_true, y_pred, sample_weight = weights),
        "MSE": mean_squared_error(y_true, y_pred, sample_weight = weights),
    }

def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))

def difference_list(list1, list2):
    return list(set(list1)- set(list2))

def union_list(list1, list2):
    return list(set(list1).union(set(list2)))

def reduce_memory_usage_pl(df: pl.DataFrame) -> pl.DataFrame:
    """ Reduce memory usage by polars dataframe {df} with name {name} by changing its data types."""
    
    print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
    Numeric_Float_types = [pl.Float32,pl.Float64]
    for col in df.columns:
        if col in ['date_id', 'time_id', 'symbol_id', 'weight']:
            continue

        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        # Ignore columns with all nulls
        if not c_min: continue

        # Casting
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))
            else:
                pass
        #elif col_type == pl.Utf8:
        #    df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass

    print(f"Memory usage after optimization is: {round(df.estimated_size('mb'), 2)} MB")
    return df

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)!='category':
            c_min,c_max = df[col].min(),df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)  
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    return df


def check_nulls(df: pl.DataFrame):
    null_columns = df.select(
        [pl.col(col).is_null().any().alias(col) for col in df.columns]
    ).row(0)  # Get the first row, which contains True/False for each column
    null_columns = [col for col, has_nulls in zip(df.columns, null_columns) if has_nulls]
    if len(null_columns)>0: print(null_columns)
    #######
    # Check for null counts for each column
    null_counts = (
        df.select(
            [pl.col(col).is_null().sum().alias(col) for col in df.columns]
        )
        .transpose(include_header=True).filter(pl.col("column_0") > 0)
    )
    # Convert to a dictionary or DataFrame for viewing
    null_summary = null_counts.rename({"column_0": "null_count"})
    if len(null_columns)>0: print(null_summary)