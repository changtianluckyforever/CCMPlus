import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def app_sum(x):
    return x.iloc[:, -1440:].sum()


def agg_files():
    all_df = None
    for i in range(1, 15):
        d = f"0{i}" if i < 10 else i
        filename = f"azurefunctions-dataset2019/invocations_per_function_md.anon.d{d}.csv"
        print(filename)
        df = pd.read_csv(filename)
        ts = df.groupby(["HashApp"]).apply(app_sum).reset_index()
        if all_df is None:
            all_df = ts
        else:
            all_df = pd.merge(all_df, ts, on="HashApp", how="left", suffixes=("", f"_{d}"))

    # all_df.to_csv("processed_data/azure_app.csv")

    new_df = {}
    for i, row in all_df.iterrows():
        new_df[row["HashApp"]] = row[1:].values

    new_df = pd.DataFrame(new_df)

    new_df.to_csv("processed_data/azure_app_T.csv")


def agg_hourly():
    date_range = pd.date_range('2019-07-15 00:00:00', '2019-07-28 23:59:00', freq='1T', tz="UTC")
    df = pd.read_csv("processed_data/azure_app_T.csv", index_col=0)
    print(len(date_range), df.shape)
    df.index = date_range
    hourly_df = df.resample(rule="1H").sum()
    hourly_df = pd.DataFrame(hourly_df, dtype=np.int32)
    hourly_df.index.name = "time"
    print(hourly_df.index)
    print(len(hourly_df))
    hourly_df.to_csv("processed_data/azure_app_H.csv")


agg_hourly()
