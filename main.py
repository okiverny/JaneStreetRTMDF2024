import os
import numpy as np
import pandas as pd
import polars as pl
import time

def main():

    # Read data
    for data_id in [9]: 
        data=pl.read_parquet(f"/Users/okiverny/workspace/Kaggle/JaneStreetRTMDF2024/data/partition_id={data_id}/part-0.parquet")

        # Column names
        lags_cols = [f"responder_{idx}_lag_1" for idx in range(9)]
        responder_cols = [f"responder_{idx}" for idx in range(9)]
        rename_dict = {responder_col:lags_col for lags_col,responder_col in zip(lags_cols, responder_cols)}

        # iterate over date_id
        first_date_id = data["date_id"].unique(maintain_order=True).first()
        for date_id in data["date_id"].unique(maintain_order=True):

            if date_id>1532: continue

            data_daily = data.filter(pl.col("date_id")==date_id)

            if date_id>first_date_id:
                data_lags = data.filter(pl.col("date_id")==date_id-1).select(["date_id","time_id","symbol_id"] + responder_cols).rename(rename_dict)
            else:
                continue

            # iterate over time_id
            for time_id in data_daily["time_id"].unique(maintain_order=True):
                provided_data = data_daily.filter(pl.col("time_id")==time_id)
                provided_lags = data_lags if time_id==0 else None

                print(50*'=')
                print(date_id, time_id)
                print(provided_data)
                print(provided_lags)

if __name__ == "__main__":
    print('Running ...')
    main()