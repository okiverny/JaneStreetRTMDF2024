import os
import numpy as np
import pandas as pd
import polars as pl
import time
import optuna
import pickle
import joblib

import lightgbm as lgbm
from  lightgbm import LGBMRegressor

from DataLoader import SymbolLagsCollection, RetrainDataCollection
from Configs import (
    MissingValueConfig,
    FeatureConfig,
    ModelConfig,
    LagFeaturesConfig,
    MLForecast
)
from utils import calculate_metrics
from optimization import run_optuna

def process_train(collection_size: int, missing_config: MissingValueConfig | None):

    ################################ Actual Work: Init #################################

    date_buffer_size = collection_size
    symbol_lags_collection = SymbolLagsCollection(
        date_buffer_size=date_buffer_size,
        missing_config=missing_config
    )
    retrain_data_collection = RetrainDataCollection(date_buffer_size=date_buffer_size)
    
    ############################## END Actual Work: Init ###############################

    # Collected and processed data and periodically save to file
    processed_data = pl.DataFrame()

    # Read data
    for data_file_index, data_id in enumerate([9]):  # 4,5,6,7,8,9
        data=pl.read_parquet(f"/Users/okiverny/workspace/Kaggle/JaneStreetRTMDF2024/data/partition_id={data_id}/part-0.parquet")
        print(f'This file contains {len(data["date_id"].unique(maintain_order=True))} days of data')

        # Column names
        lags_cols = [f"responder_{idx}_lag_1" for idx in range(9)]
        responder_cols = [f"responder_{idx}" for idx in range(9)]
        rename_dict = {responder_col:lags_col for lags_col, responder_col in zip(lags_cols, responder_cols)}

        # Get first and last date_id in this file
        first_date_id = data["date_id"].unique(maintain_order=True).to_list()[0]
        last_date_id = data["date_id"].unique(maintain_order=True).to_list()[-1]

        # iterate over date_id
        for date_id in data["date_id"].unique(maintain_order=True):

            if date_id>1538: continue

            data_daily = data.filter(pl.col("date_id")==date_id)

            if (date_id==first_date_id) and (data_file_index==0):
                continue
            else:
                data_lags = data.filter(pl.col("date_id")==date_id-1).select(["date_id","time_id","symbol_id"] + responder_cols).rename(rename_dict)

            # collect the combined data from 'today'
            today_processed_data = []

            # iterate over time_id
            for time_id in data_daily["time_id"].unique(maintain_order=True):
                provided_data = data_daily.filter(pl.col("time_id")==time_id)
                provided_lags = data_lags.clone() if time_id==0 else None

                ################################ Actual Work #################################
                start = time.time()

                # fill missing values in features of test data
                if missing_config!=None:
                    provided_data = missing_config.impute_missing_values(provided_data)

                # Backup the features for online learning
                retrain_data_collection.add_data(provided_data.drop(responder_cols))

                if provided_lags is not None:
                    # Add new lags data to collection
                    symbol_lags_collection.add_lags(provided_lags)

                    combined_lagged_data = []
                    new_symbols_appeared = []
                    combined_lagged_target = []

                    # Construct features for each 'symbol_id' in test data
                    for symbol_id in provided_data['symbol_id']:

                        if len(symbol_lags_collection.get_lags(symbol_id))==0:
                            print(f'No lags for symbol_id={symbol_id} is available! Skipping computation of lagged and temporal features')
                            new_symbols_appeared.append(symbol_id)
                        else:
                            df = symbol_lags_collection.construct_features(symbol_lags_collection.get_lags(symbol_id=symbol_id))

                            # Backup the targets from yesterday
                            lagged_target = df.with_columns(
                                pl.col('responder_6_lag_1').alias('responder_6') # rename the target column
                            ).select(
                                ['date_id', 'time_id', 'symbol_id', 'responder_6']
                            ).filter(pl.col("date_id")==date_id-1) # select yesterday's targets
                            combined_lagged_target.append(lagged_target)

                            # Increment the 'date_id' column by 1 and keep only the values from 'today'
                            df = df.with_columns(
                                (pl.col("date_id") + 1).alias("date_id")  # Increment values and keep the same column name
                            ).filter(pl.col("date_id")==date_id)

                            # Append to the list
                            combined_lagged_data.append(df)

                    # Combining the lagged data from all symbol_id
                    combined_lagged_data = pl.concat(combined_lagged_data)
                    combined_lagged_target = pl.concat(combined_lagged_target)

                    # Backup the lagged targets for online training
                    retrain_data_collection.add_lagged_target(combined_lagged_target)

                    print('---------')
                    print('date=', date_id)
                    print(retrain_data_collection.retrain_data)
                    print(retrain_data_collection.retrain_lagged_target)

                    # Check the processing time
                    end = time.time()
                    print(f"time: {end - start:.4f} seconds for time_id={time_id}")

                # Join test data and lagged data
                processed_data_time_id = provided_data.join(combined_lagged_data, on=['date_id', 'time_id', 'symbol_id'],  how='left')


                # Append the processed data from today's date_id
                today_processed_data.append(processed_data_time_id)


                # Measuring the running time
                end = time.time()
                #print(f"time: {end - start:.4f} seconds for time_id={time_id}")
                ################################ END Actual Work #################################

            # Concatenate the today's processed data with the data from previous days
            processed_data = pl.concat([processed_data]+today_processed_data)
            print(date_id, (date_id+1) % 60,  (date_id+1) // 60)
            #print(processed_data)

            ################################ Check Nulls ################################
            #check_nulls(processed_data)
            ################################ Check Nulls ################################

            # Store a portion of data with 60 days
            if ((date_id+1) % 60 == 0) or (data_id==9 and date_id==last_date_id):
                print(f'Saving a partinion of processed data ({ len(processed_data["date_id"].unique()) } days)')
                if not os.path.exists('processed_data/'): os.makedirs('processed_data/')
                processed_data.write_parquet(f'processed_data/data_part_{(date_id+1) // 60}.parquet')
                processed_data = pl.DataFrame()

    print('Processing of the train data is finished!')


def evaluate_model(feature_config: FeatureConfig, model_config: ModelConfig, train_dates: list, test_dates: list, use_weights: bool = False):
    dataframes_train, dataframes_test = [], []
    for data_part in [26,27,28]:
        data=pl.read_parquet(f"processed_data/data_part_{data_part}.parquet").filter( (pl.col("date_id")>=train_dates[0]) & (pl.col("date_id")<train_dates[1]) )
        dataframes_train.append(data)

        data_test=pl.read_parquet(f"processed_data/data_part_{data_part}.parquet").filter( (pl.col("date_id")>test_dates[0]) & (pl.col("date_id")>=test_dates[1]) )
        dataframes_test.append(data_test)

    # Manipulate with the train data
    train_data = pl.concat(dataframes_train)
    del dataframes_train

    # print('Columns in the train data frame:')
    # for col,col_type in zip(train_data.columns, train_data.dtypes):
    #     print(col,' - ',col_type)

    #print( train_data )
    #print(train_data.null_count())
    #print(data["symbol_id"].unique(maintain_order=True).to_list())

    train_features, train_target, train_original_target, train_weight = feature_config.get_X_y(
        train_data, categorical=True, exogenous=True
    )
    del train_data

    # Manipulate with the test data
    test_data = pl.concat(dataframes_test)
    del dataframes_test

    test_features, test_target, test_original_target, test_weight = feature_config.get_X_y(
        test_data, categorical=True, exogenous=True
    )
    del test_data

    # Construct the MLForecast instance
    ml_model = MLForecast(
        model_config=model_config,
        feature_config=feature_config,
        missing_config=None
    )

    # Use sample weights in the fitting?
    train_weight = train_weight if use_weights else None
    test_weight = test_weight if use_weights else None

    # Add feature names and indicate categorical features to LightGBM
    if type(model_config.model)==LGBMRegressor:
        fit_kwargs={
            'feature_name': train_features.columns,
            'categorical_feature': feature_config.categorical_features,
        }
    else:
        fit_kwargs={}

    # Fit the model on the train data
    ml_model.fit(train_features, train_target, w=train_weight, fit_kwargs=fit_kwargs)

    y_pred_train = ml_model.predict(train_features)
    y_pred = ml_model.predict(test_features)

    # Extract the feature importance
    feat_importance_df = ml_model.feature_importance()
    print(feat_importance_df)

    # Calculate various metrics in the test data
    metrics_train = calculate_metrics(train_target, y_pred_train, model_config.name, train_weight)
    metrics_test = calculate_metrics(test_target, y_pred, model_config.name, test_weight)
    print('Train metrics:', metrics_train)
    print('Test metrics:', metrics_test)

    return y_pred, metrics_test, feat_importance_df



def main():
    missing_config = MissingValueConfig(
        bfill_columns=[],
        ffill_columns=[f"feature_{idx:02d}" for idx in range(79)],
        zero_fill_columns=[f"feature_{idx:02d}" for idx in range(79)],
        fill_symbols_daily=True
    )
    missing_config=None

    print('Processing train data!')
    process_train(5, missing_config)


    # ##################
    # feature_config = FeatureConfig(
    #     date='time_id',
    #     target='responder_6',
    #     weight='weight',
    #     continuous_features=[f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11]] + \
    #         [f"responder_{idx}_lag_1" for idx in range(9)] + \
    #         ['time_id_Elapsed',
    #          'time_id_Period_968_sin_1','time_id_Period_968_sin_2','time_id_Period_968_sin_3','time_id_Period_968_sin_4','time_id_Period_968_sin_5',
    #          'time_id_Period_968_cos_1','time_id_Period_968_cos_2','time_id_Period_968_cos_3','time_id_Period_968_cos_4','time_id_Period_968_cos_5'],
    #     categorical_features=['symbol_id'] + [f"feature_{idx:02d}" for idx in [9, 10, 11]] + [f"time_id_Period_{period}" for period in [332, 725, 968]],
    #     boolean_features=[],
    #     index_cols=["time_id"],
    #     exogenous_features=[f"feature_{idx:02d}" for idx in range(79)]
    # )

    # lgb_params={
    #     "boosting_type": "gbdt",
    #     "objective":        'mean_squared_error',
    #     "random_state":  2025,
    #     "max_depth":     5,
    #     "learning_rate": 0.05,
    #     "n_estimators":  160,
    #     "colsample_bytree": 0.6,
    #     "colsample_bynode": 0.6,
    #     "reg_alpha":     0.2,
    #     "reg_lambda":    5,
    #     "extra_trees":  True,
    #     "num_leaves":    64,
    #     "max_bin":       255,
    #     #'device':'gpu',
    #     "n_jobs":        -1,
    #     #"verbose":       1,
    # }

    # model_config = ModelConfig(
    #     model=LGBMRegressor(**lgb_params), # try MAE or Huber loss to penalize outliers less!!
    #     name="LightGBM",
    #     normalize=False, # LGBM is not affected by normalization
    #     fill_missing=False, # # LGBM handles missing values
    # )

    # evaluate_model(feature_config, model_config, train_dates=[1540, 1600], test_dates=[1600, 1640], use_weights=True)

    ##################### Hyperparamater optimization #####################
    #run_optuna(3)
    ##################### END Hyperparamater optimization #####################



if __name__ == "__main__":
    print('Running ...')
    main()
