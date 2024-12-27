import os
import numpy as np
import pandas as pd
import polars as pl
import time

from DataLoader import SymbolLagsCollection
from autoregressive_features import (
    add_lags,
    add_rolling_features,
    add_seasonal_rolling_features,
    add_ewma
)
from temporal_features import (
    add_temporal_features,
    bulk_add_fourier_features
)

def main():

    ################################ Actual Work: Init #################################
    date_buffer_size = 5
    #lags_collection = LagsCollection(date_buffer_size)
    symbol_lags_collection = SymbolLagsCollection(
        date_buffer_size=date_buffer_size,
        lags=[967*(i+1) for i in range(2)]
    )
    ############################## END Actual Work: Init ###############################

    # Collected and processed data and periodically save to file
    processed_data = pl.DataFrame()

    # Read data
    for data_file_index, data_id in enumerate([8, 9]):
        data=pl.read_parquet(f"/Users/okiverny/workspace/Kaggle/JaneStreetRTMDF2024/data/partition_id={data_id}/part-0.parquet")
        print(f'This file contains {len(data["date_id"].unique(maintain_order=True))} days of data')

        # Column names
        lags_cols = [f"responder_{idx}_lag_1" for idx in range(9)]
        responder_cols = [f"responder_{idx}" for idx in range(9)]
        rename_dict = {responder_col:lags_col for lags_col, responder_col in zip(lags_cols, responder_cols)}

        # iterate over date_id
        first_date_id = data["date_id"].unique(maintain_order=True).to_list()[0]
        last_date_id = data["date_id"].unique(maintain_order=True).to_list()[-1]
        for date_id in data["date_id"].unique(maintain_order=True):

            #if date_id>1538: continue

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

                # if provided_lags is not None:
                #     print(50*'=')
                #     print(date_id, time_id)
                #     print('Provided Data:')
                #     print(provided_data)
                #     print(provided_lags)

                ################################ Actual Work #################################
                start = time.time()

                if provided_lags is not None:
                    #lags_collection += provided_lags
                    symbol_lags_collection.add_lags(provided_lags)


                    #print('symbol_lags_collection')
                    #print(symbol_lags_collection.symbol_data.keys)
                    #print( [i for i in symbol_lags_collection.symbol_data] )
                    #print(len(symbol_lags_collection))

                    combined_lagged_data = []

                    # Construct features for each 'symbol_id' in test data
                    for symbol_id in provided_data['symbol_id']:
                        #print('====================== Processing symbol_id:', symbol_id)
                        if len(symbol_lags_collection.get_lags(symbol_id))==0:
                            print(f'No lags for symbol_id={symbol_id} is available! Skipping computation of lagged and temporal features')

                        else:
                            #print(100*'===')
                            df, lag_features = add_lags(symbol_lags_collection.get_lags(symbol_id=symbol_id), [968*(i+1) for i in range(2)], 'responder_6_lag_1')
                            #print(df)
                            #print('lag_features', lag_features)

                            #print(100*'===')
                            df, rolling_features = add_rolling_features(df, rolls=[16, 60, 120], column='responder_6_lag_1', agg_funcs=["mean", "std"], n_shift=0, use_32_bit=True)
                            #print(df)
                            #print('rolling_features:', rolling_features)

                            #print(100*'===')
                            df, season_rolling_features = add_seasonal_rolling_features(df, seasonal_periods=[968, 2*968], rolls=[3], column='responder_6_lag_1', agg_funcs=["mean", "std"], n_shift=0, use_32_bit=True)
                            #print(df)
                            #print('season_rolling_features:', season_rolling_features)

                            #print(100*'===')
                            df, ewma_features = add_ewma(df, 'responder_6_lag_1', spans=[968, 4*968], n_shift=0, use_32_bit=True)
                            #print(df)
                            #print('ewma_features:', ewma_features)

                            # TODO: Move out of symbol_id loop ?
                            #print(100*'===')
                            df, temporal_features_time = add_temporal_features(df, 'time_id', periods=[8, 16, 242], add_elapsed=True, drop=False, use_32_bit=True)
                            df, temporal_features_date = add_temporal_features(df, 'date_id', periods=[5], add_elapsed=True, drop=False, use_32_bit=True)
                            #print(df)
                            #print('temporal_features_time:', temporal_features_time)
                            #print('temporal_features_date:', temporal_features_date)

                            #print(100*'===')
                            df, fourier_features = bulk_add_fourier_features(df, columns_to_encode=['time_id_Period_16', 'date_id_Period_5'], max_values=[16, 5], n_fourier_terms=3, use_32_bit=True)
                            #print(df)
                            #print('fourier_features:', fourier_features)

                            #print('The resulting lag data frame:')
                            #print(df)

                            # Increment the 'date_id' column by 1 and keep only the values from 'today'
                            #print(f'Today is {date_id}')
                            df = df.with_columns(
                                (pl.col("date_id") + 1).alias("date_id")  # Increment values and keep the same column name
                            ).filter(pl.col("date_id")==date_id)
                            #print(df)

                            # Append to the list
                            combined_lagged_data.append(df)

                    # Combining the lagged data from all symbol_id
                    combined_lagged_data = pl.concat(combined_lagged_data)

                    # Check the processing time
                    end = time.time()
                    print(f"time: {end - start:.4f} seconds for time_id={time_id}")


                # Join test data and lagged data
                processed_data_time_id = provided_data.join(combined_lagged_data, on=['date_id', 'time_id', 'symbol_id'],  how='left')
                #print(processed_data_time_id)
                #print(processed_data_time_id.columns)

                # Append the processed data from today's date_id
                today_processed_data.append(processed_data_time_id)



                # Measuring the running time
                end = time.time()
                #print(f"time: {end - start:.4f} seconds for time_id={time_id}")
                ################################ END Actual Work #################################

            # Concatenate the today's processed data with the data from previous days
            processed_data = pl.concat([processed_data]+today_processed_data)
            print(date_id, (date_id+1) % 60,  (date_id+1) // 60)
            print(processed_data)

            # Store a portion of data with 60 days
            if ((date_id+1) % 60 == 0) or (data_id==9 and date_id==last_date_id):
                print('Saving a partinion of processed data (60 days)')
                processed_data.write_parquet(f'processed_data/data_part_{(date_id+1) // 60}.parquet')
                processed_data = pl.DataFrame()


if __name__ == "__main__":
    print('Running ...')
    main()