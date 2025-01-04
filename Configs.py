import copy
from dataclasses import MISSING, dataclass, field
from typing import List, Union, Tuple
from sklearn.base import BaseEstimator, clone
from autoregressive_features import (
    add_lags,
    add_rolling_features,
    add_seasonal_rolling_features,
    add_ewma
)

import polars as pl

@dataclass
class LagFeaturesConfig:
    lag_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be lagged with values from 'lag_values' list.  The tag '_lag_N' will be added."},
    )
    lag_values: List[int] = field(
        default_factory=list,
        metadata={"help": "Columns from 'lag_columns' to be lagged by these values. The tag '_lag_N' will be added."},
    )

    rolling_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be rolled"},
    )
    rolling_values: List[int] = field(
        default_factory=list,
        metadata={"help": "Rolling windows."},
    )
    rolling_agg_funcs: List[str] = field(
        default_factory=list,
        metadata={"help": "Rolling functions."},
    )

    seasonal_rolling_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be rolled"},
    )
    seasonal_periods: List[int] = field(
        default_factory=list,
        metadata={"help": "Seasonal Rolling periods."},
    )
    seasonal_rolls: List[int] = field(
        default_factory=list,
        metadata={"help": "Rolling windows."},
    )
    seasonal_rolling_agg_funcs: List[str] = field(
        default_factory=list,
        metadata={"help": "Rolling functions."},
    )


    def create_lag_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        # Example:
        # lags_config = LagFeaturesConfig(
        #     lag_columns=['responder_6_lag_1'],
        #     lag_values=[968*(i+1) for i in range(2)],
        #     rolling_columns=['responder_6_lag_1'],
        #     rolling_values=[120, 484],
        #     rolling_agg_funcs=["mean", "std"],
        #     seasonal_rolling_columns=['responder_6_lag_1'],
        #     seasonal_periods=[968],
        #     seasonal_rolls=[3],
        #     seasonal_rolling_agg_funcs=["mean", "std"],
        # )

        # Add lags features
        lag_features = []
        for column in self.lag_columns:
            df, new_features = add_lags(df, self.lag_values, column)
            lag_features.append(new_features)

        # Add rolling features
        rolling_features = []
        for column in self.rolling_columns:
            df, new_features = add_rolling_features(df, rolls=self.rolling_values, column=column, agg_funcs=self.rolling_agg_funcs, n_shift=0, use_32_bit=True)
            rolling_features.append(new_features)

        # Add seasonal rolling features
        seasonal_rolling_features = []
        for column in self.seasonal_rolling_columns:
            df, new_features = add_seasonal_rolling_features(df, seasonal_periods=self.seasonal_periods, rolls=self.seasonal_rolls, column=column, agg_funcs=self.seasonal_rolling_agg_funcs, n_shift=0, use_32_bit=True)
            seasonal_rolling_features.append(new_features)

        # More to be added ...

        return (df, lag_features+rolling_features)


@dataclass
class MissingValueConfig:
    bfill_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be filled using strategy=`bfill`"},
    )
    ffill_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be filled using strategy=`ffill`"},
    )
    zero_fill_columns: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names to be filled with 0"},
    )

    fill_symbols_daily: bool = field(
        default_factory=bool,
        metadata={"help": "A Boolean flag for filling missing data for each symbol_id in the collection deque."},
    )

    def impute_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Impute missing values in the DataFrame according to specified strategies.
        """
        # Create a copy to avoid mutating the original DataFrame
        df = df.clone()

        # Apply backward fill (bfill) for specified columns
        if self.bfill_columns:
            for col in self.bfill_columns:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).fill_null(strategy="backward").alias(col)
                    )

        # Apply forward fill (ffill) for specified columns
        if self.ffill_columns:
            for col in self.ffill_columns:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).fill_null(strategy="forward").alias(col)
                    )

        # Fill specified columns with 0
        if self.zero_fill_columns:
            for col in self.zero_fill_columns:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).fill_null(0).alias(col)
                    )

        # Check for remaining null values and handle them
        null_columns = [
            col for col in df.columns if df.select(pl.col(col).is_null().any()).row(0)[0]
        ]

        # Separate numeric and non-numeric columns for further handling
        numeric_cols = [col for col in null_columns if df.schema[col] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
        non_numeric_cols = [col for col in null_columns if col not in numeric_cols]

        # Fill remaining numeric columns with their mean
        for col in numeric_cols:
            df = df.with_columns(
                pl.col(col).fill_null(pl.col(col).mean()).alias(col)
            )

        # Fill remaining non-numeric columns with "NA"
        for col in non_numeric_cols:
            df = df.with_columns(
                pl.col(col).fill_null("NA").alias(col)
            )

        return df

@dataclass
class FeatureConfig:
    date: List[str] = field(
        default=MISSING,
        metadata={"help": "Column name of the date column"},
    )
    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )
    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in case of transformed target. If None, it will be assigned the same value as target"
        },
    )
    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields. Defaults to []"},
    )
    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )
    index_cols: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names that need to be set as index in the X and Y dataframes."
        },
    )
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )
    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at least one feature defined in categorical or continuous columns"
        self.feature_list = (
            self.categorical_features + self.continuous_features + self.boolean_features
        )
        assert (
            self.target not in self.feature_list
        ), f"`target`({self.target}) should not be present in either categorical, continuous, or boolean feature list"
        assert (
            self.date not in self.feature_list
        ), f"`date`({self.date}) should not be present in either categorical, continuous, or boolean feature list"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (
            len(extra_exog) == 0
        ), f"These exogenous features are not present in feature list: {extra_exog}"
        intersection = (
            set(self.continuous_features)
            .intersection(self.categorical_features + self.boolean_features)
            .union(
                set(self.categorical_features).intersection(
                    self.continuous_features + self.boolean_features
                )
            )
            .union(
                set(self.boolean_features).intersection(
                    self.continuous_features + self.categorical_features
                )
            )
        )
        assert (
            len(intersection) == 0
        ), f"There should not be any overlaps between the categorical, continuous, and boolean features. {intersection} are present in more than one definition"
        if self.original_target is None:
            self.original_target = self.target

    def get_X_y(
        self,
        df: pl.DataFrame,
        categorical: bool = False,
        exogenous: bool = False,
    ):
        feature_list = copy.deepcopy(self.continuous_features)
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
        if not exogenous:
            feature_list = list(set(feature_list) - set(self.exogenous_features))
        feature_list = list(set(feature_list))
        delete_index_cols = list(set(self.index_cols) - set(self.feature_list))

        # Extract X, y, and y_orig
        X = df.select(
            pl.col(feature_list + self.index_cols).exclude(delete_index_cols)
        )
        y = (
            df.select(pl.col([self.target] + self.index_cols))
            if self.target in df.columns
            else None
        )
        y_orig = (
            df.select(pl.col([self.original_target] + self.index_cols))
            if self.original_target in df.columns
            else None
        )

        return X, y, y_orig


@dataclass
class ModelConfig:

    model: BaseEstimator = field(
        default=MISSING, metadata={"help": "Sci-kit Learn Compatible model instance"}
    )

    name: str = field(
        default=None,
        metadata={
            "help": "Name or identifier for the model. If left None, will use the string representation of the model"
        },
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Flag whether to normalize the input or not"},
    )
    fill_missing: bool = field(
        default=True,
        metadata={"help": "Flag whether to fill missing values before fitting"},
    )
    encode_categorical: bool = field(
        default=False,
        metadata={"help": "Flag whether to encode categorical values before fitting"},
    )
    categorical_encoder: BaseEstimator = field(
        default=None,
        metadata={"help": "Categorical Encoder to be used"},
    )

    def __post_init__(self):
        assert not (
            self.encode_categorical and self.categorical_encoder is None
        ), "`categorical_encoder` cannot be None if `encode_categorical` is True"

    def clone(self):
        self.model = clone(self.model)
        return self