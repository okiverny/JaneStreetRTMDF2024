import copy
from dataclasses import MISSING, dataclass, field
from typing import List, Union

import polars as pl


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