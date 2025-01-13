import copy
import warnings
import polars as pl
import numpy as np
from dataclasses import MISSING, dataclass, field
from typing import List, Union, Tuple, Dict
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from autoregressive_features import (
    add_lags,
    add_rolling_features,
    add_seasonal_rolling_features,
    add_ewma
)
from utils import intersect_list, difference_list
from utils import R2_metric, R2w_metric

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
    date: str = field(
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
    weight: str = field(
        default=None,
        metadata={"help": "Column name of the weight column"},
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

        # Extract X, y, y_orig and w
        X = df.select(
            pl.col(feature_list + self.index_cols).exclude(delete_index_cols)
        )
        y = (
            df.select(pl.col([self.target])).to_series() # + self.index_cols
            if self.target in df.columns
            else None
        )
        y_orig = (
            df.select(pl.col([self.original_target])).to_series() # + self.index_cols
            if self.original_target in df.columns
            else None
        )
        w = (
            df.select(pl.col([self.weight])).to_series() # + self.index_cols
            if self.weight in df.columns
            else None
        )

        return X, y, y_orig, w


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


class MLForecast:
    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: FeatureConfig,
        missing_config: MissingValueConfig = None,
        target_transformer: object = None,
    ) -> None:
        """Convenient wrapper around scikit-learn style estimators

        Args:
            model_config (ModelConfig): Instance of the ModelConfig object defining the model
            feature_config (FeatureConfig): Instance of the FeatureConfig object defining the features
            missing_config (MissingValueConfig, optional): Instance of the MissingValueConfig object
                defining how to fill missing values. Defaults to None.
            target_transformer (object, optional): Instance of target transformers from src.transforms.
                Should support `fit`, `transform`, and `inverse_transform`. It should also
                return `pd.Series` with datetime index to work without an error. Defaults to None.
        """
        self.model_config = model_config
        self.feature_config = feature_config
        self.missing_config = missing_config
        self.target_transformer = target_transformer
        # self._model = clone(model_config.model)
        self._model = model_config.model
        if self.model_config.normalize:
            self._scaler = StandardScaler()
        if self.model_config.encode_categorical:
            self._cat_encoder = self.model_config.categorical_encoder
            self._encoded_categorical_features = copy.deepcopy(
                self.feature_config.categorical_features
            )

    def fit(
        self,
        X: pl.DataFrame,
        y: Union[pl.Series, np.ndarray],
        w: Union[pl.Series, np.ndarray],
        is_transformed: bool = False,
        fit_kwargs: Dict = {},
    ):
        """Handles standardization, missing value handling, and training the model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns
            y (Union[pd.Series, np.ndarray]): Dataframe, Series, or np.ndarray with the targets
            is_transformed (bool, optional): Whether the target is already transformed.
            If `True`, fit wont be transforming the target using the target_transformer
                if provided. Defaults to False.
            fit_kwargs (Dict, optional): The dictionary with keyword args to be passed to the
                fit funciton of the model. Defaults to {}.
        """
        missing_feats = difference_list(X.columns, self.feature_config.feature_list)
        if len(missing_feats) > 0:
            warnings.warn(
                f"Some features in defined in FeatureConfig is not present in the dataframe. Ignoring these features: {missing_feats}"
            )
        self._continuous_feats = intersect_list(
            self.feature_config.continuous_features, X.columns
        )
        self._categorical_feats = intersect_list(
            self.feature_config.categorical_features, X.columns
        )
        self._boolean_feats = intersect_list(
            self.feature_config.boolean_features, X.columns
        )
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)

        if self.model_config.encode_categorical:
            missing_cat_cols = difference_list(
                self._categorical_feats,
                self._cat_encoder.cols,
            )
            assert (
                len(missing_cat_cols) == 0
            ), f"These categorical features are not handled by the categorical_encoder: {missing_cat_cols}"
            
            # Fit the encoder before getting feature names
            X = self._cat_encoder.fit_transform(X, y)
            
            # Now get the feature names from the fitted encoder
            try:
                feature_names = self._cat_encoder.get_feature_names()
            except AttributeError:
                # For newer versions of sklearn
                feature_names = self._cat_encoder.get_feature_names_out()
            
            self._encoded_categorical_features = difference_list(
                feature_names,
                self.feature_config.continuous_features + self.feature_config.boolean_features,
            )
        else:
            self._encoded_categorical_features = []


        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.fit_transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        self._train_features = X.columns

        if not is_transformed and self.target_transformer is not None:
            y = self.target_transformer.fit_transform(y)

        self._model.fit(X, y, sample_weight=w, **fit_kwargs)

        return self

    def update(
        self,
        X: pl.DataFrame,
        y: Union[pl.Series, np.ndarray],
        w: Union[pl.Series, np.ndarray],
        is_transformed: bool = False,
        fit_kwargs: Dict = {},
    ):
        """Updates a model which was already trained and handles standardization and missing values

        Args:
            X (pd.DataFrame): The dataframe with the features as columns
            y (Union[pd.Series, np.ndarray]): Dataframe, Series, or np.ndarray with the targets
            is_transformed (bool, optional): Whether the target is already transformed.
            If `True`, fit wont be transforming the target using the target_transformer
                if provided. Defaults to False.
            fit_kwargs (Dict, optional): The dictionary with keyword args to be passed to the
                fit funciton of the model. Defaults to {}.
        """
        missing_feats = difference_list(X.columns, self.feature_config.feature_list)
        if len(missing_feats) > 0:
            warnings.warn(
                f"Some features in defined in FeatureConfig is not present in the dataframe. Ignoring these features: {missing_feats}"
            )
        self._continuous_feats = intersect_list(
            self.feature_config.continuous_features, X.columns
        )
        self._categorical_feats = intersect_list(
            self.feature_config.categorical_features, X.columns
        )
        self._boolean_feats = intersect_list(
            self.feature_config.boolean_features, X.columns
        )
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)

        if self.model_config.encode_categorical:
            missing_cat_cols = difference_list(
                self._categorical_feats,
                self._cat_encoder.cols,
            )
            assert (
                len(missing_cat_cols) == 0
            ), f"These categorical features are not handled by the categorical_encoder: {missing_cat_cols}"

            # Fit the encoder before getting feature names
            X = self._cat_encoder.fit_transform(X, y)

            # Now get the feature names from the fitted encoder
            try:
                feature_names = self._cat_encoder.get_feature_names()
            except AttributeError:
                # For newer versions of sklearn
                feature_names = self._cat_encoder.get_feature_names_out()

            self._encoded_categorical_features = difference_list(
                feature_names,
                self.feature_config.continuous_features + self.feature_config.boolean_features,
            )
        else:
            self._encoded_categorical_features = []


        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )
        self._train_features = X.columns

        if not is_transformed and self.target_transformer is not None:
            y = self.target_transformer.transform(y)

        self._model.fit(X, y, sample_weight=w, init_model=self._model, **fit_kwargs)

        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Predicts on the given dataframe using the trained model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns. The index is passed on to the prediction series

        Returns:
            pd.Series: predictions using the model as a pandas Series with datetime index
        """
        # Quick workaround:
        self._train_features = X.columns

        assert len(intersect_list(self._train_features, X.columns)) == len(
            self._train_features
        ), f"All the features during training is not available while predicting: {difference_list(self._train_features, X.columns)}"
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            X = self._cat_encoder.transform(X)
        if self.model_config.normalize:
            X[
                self._continuous_feats + self._encoded_categorical_features
            ] = self._scaler.transform(
                X[self._continuous_feats + self._encoded_categorical_features]
            )

        y_pred = pl.Series(
            name=f"{self.model_config.name}",
            values=self._model.predict(X).ravel(),
        )
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_pred)
            y_pred.name = f"{self.model_config.name}"
        return y_pred

    def feature_importance(self) -> pl.DataFrame:
        """Generates the feature importance dataframe, if available. For linear
            models the coefficients are used and tree based models use the inbuilt
            feature importance. For the rest of the models, it returns an empty dataframe.

        Returns:
            pl.DataFrame: Feature Importance dataframe, sorted in descending order of its importances.
        """
        if hasattr(self._model, "coef_") or hasattr(self._model, "feature_importances_"):
            # Determine the importance values
            importances = (
                self._model.coef_.ravel()
                if hasattr(self._model, "coef_")
                else self._model.feature_importances_.ravel()
            )
            feat_df = pl.DataFrame({
                "feature": self._train_features,
                "importance": importances
            })

            # Add absolute importance column for sorting
            feat_df = feat_df.with_columns(
                pl.col("importance").abs().alias("_abs_imp")
            )

            # Sort by absolute importance and drop the temporary column
            feat_df = feat_df.sort("_abs_imp", descending=True).drop("_abs_imp")

        else:
            # Return an empty Polars DataFrame if no feature importance is available
            feat_df = pl.DataFrame()

        return feat_df