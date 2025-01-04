import os
import numpy as np
import pandas as pd
import polars as pl
import time

from DataLoader import SymbolLagsCollection
from Configs import (
    MissingValueConfig,
    FeatureConfig,
    LagFeaturesConfig
)
from utils import check_nulls

def main():

    ################################ Actual Work: Init #################################
    # missing_config = MissingValueConfig(
    #     bfill_columns=[],
    #     ffill_columns=[f"feature_{idx:02d}" for idx in range(79)],
    #     zero_fill_columns=[f"feature_{idx:02d}" for idx in range(79)],
    #     fill_symbols_daily=True
    # )
    missing_config=None

    date_buffer_size = 5
    symbol_lags_collection = SymbolLagsCollection(
        date_buffer_size=date_buffer_size,
        missing_config=missing_config
    )
    
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

                if provided_lags is not None:
                    # Add new lags data to collection
                    symbol_lags_collection.add_lags(provided_lags)

                    combined_lagged_data = []
                    new_symbols_appeared = []

                    # Construct features for each 'symbol_id' in test data
                    for symbol_id in provided_data['symbol_id']:

                        if len(symbol_lags_collection.get_lags(symbol_id))==0:
                            print(f'No lags for symbol_id={symbol_id} is available! Skipping computation of lagged and temporal features')
                            new_symbols_appeared.append(symbol_id)
                        else:
                            df = symbol_lags_collection.construct_features(symbol_lags_collection.get_lags(symbol_id=symbol_id))

                            # Increment the 'date_id' column by 1 and keep only the values from 'today'
                            df = df.with_columns(
                                (pl.col("date_id") + 1).alias("date_id")  # Increment values and keep the same column name
                            ).filter(pl.col("date_id")==date_id)

                            # Append to the list
                            combined_lagged_data.append(df)

                    # Combining the lagged data from all symbol_id
                    combined_lagged_data = pl.concat(combined_lagged_data)

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


if __name__ == "__main__":
    print('Running ...')
    main()
