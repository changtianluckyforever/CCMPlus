import os
import random
from datetime import datetime

import numpy as np
import pandas as pd


def ita_pre_process(trace_name, timezone=False):
    df = pd.read_csv(f"../data/internet_traffic_archive/time_series_requests/{trace_name}.tsv", header=None,
                     names=["datetime"])
    print(df.head())
    df["datetime"] = pd.to_datetime(df['datetime'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d-%H:%M:%S %z')),
                                    utc=timezone)
    df["date"] = df["datetime"].apply(lambda x: x.date())
    print(df.head())
    os.mkdir(f"../data/internet_traffic_archive/time_series_requests/daily_req/{trace_name}/")
    for gn, g in df.groupby(["date"]):
        with open(f"../data/internet_traffic_archive/time_series_requests/daily_req/{trace_name}/{gn}.tsv", "w",
                  encoding="utf-8") as f:
            f.writelines("\n".join(list(map(lambda x: str(x.timestamp()), g["datetime"]))))


def load_ita_trace(trace_name, weekday=1, days=7, relative=False):
    base_dir = "../data/internet_traffic_archive/time_series_requests/daily_req"
    if "poisson" == trace_name:
        return np.cumsum(np.random.poisson(5, int(1e7)))
    print(f"reading trace {trace_name}...")
    dir_n = f"{base_dir}/{trace_name}/"
    if not os.path.exists(dir_n):
        ita_pre_process(trace_name)
    file_dates, trg_start = [], []
    for path, dir_list, file_list in os.walk(dir_n):
        for file_name in file_list:
            date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
            if date.isoweekday() == weekday:
                trg_start.append(date)
            file_dates.append(date)
    start_day = file_dates.index(random.choice(trg_start))
    try_time = 0
    while len(file_dates) - start_day < days:
        start_day = file_dates.index(random.choice(trg_start))
        try_time += 1
        if try_time > 100:
            return
    trace = []
    for i in range(start_day, start_day + days):
        fn = f"{base_dir}/{trace_name}/{file_dates[i].strftime('%Y-%m-%d')}.tsv"
        with open(fn, "r", encoding="utf-8") as f:
            trace.append(list(map(float, f.readlines())))
    trace = np.concatenate(trace)
    if relative:
        trace -= trace[0]
    trace = pd.Series(np.ones(len(trace), dtype=np.int8), index=pd.to_datetime(trace, unit='s'))
    print(trace.head())
    return trace


def read_hourly_trace():
    trace_name = "app_rpc_filter_H"
    base_dir = "../data/processed_data/"
    df_raw = pd.read_csv(base_dir + trace_name + ".csv", index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)

    # todo Anomaly Detection
    df_raw = df_raw.rolling(window=5, min_periods=1, center=True).mean()
    # print(df_raw.head())

    # time features
    df_raw['day'] = df_raw.index.map(lambda row: row.day, 1)
    df_raw['weekday'] = df_raw.index.map(lambda row: row.weekday(), 1)
    df_raw['hour'] = df_raw.index.map(lambda row: row.hour, 1)
    # cut_hour = [-1, 5, 11, 16, 21, 23]
    # # ['last night', 'morning', 'afternoon', 'evening', 'night']
    # cut_labels = [0, 1, 2, 3, 4]
    # df_raw['hour_cut'] = pd.cut(df_raw['hour'], bins=cut_hour, labels=cut_labels)
    return df_raw


def load_hourly_trace(df_raw, num_models, days, init_days, interval="30T"):
    episode_len = (days + init_days) * 24
    selected_col = np.random.choice(df_raw.columns, num_models)
    start_idx = random.randint(0, len(df_raw) - episode_len)
    traces = df_raw.iloc[start_idx:start_idx + episode_len][selected_col].to_numpy().transpose()
    traces_ts_feat = df_raw.iloc[start_idx:start_idx + episode_len][["day", "weekday", "hour"]].to_numpy().transpose()
    if "1H" == interval:
        new_traces = traces
    elif "30T" == interval:
        split_ratio = np.random.normal(0.5, 0.1, size=(num_models, episode_len)).clip(0.4, 0.6)
        traces1 = (traces * split_ratio).astype(int)
        traces2 = traces - traces1
        new_traces = np.dstack((traces1, traces2)).reshape(num_models, episode_len * 2)
        traces_ts_feat = traces_ts_feat.repeat(2, axis=1)
    # todo Anomaly Detection
    new_traces = np.maximum(new_traces, 1)
    return new_traces, traces_ts_feat


def ita_data_analysis(trace_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(100, 5))
    df = pd.read_csv(f"../data/internet_traffic_archive/time_series_requests/{trace_name}.tsv", header=None,
                     names=["datetime"])
    print(df.head())
    df["datetime"] = pd.to_datetime(df['datetime'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d-%H:%M:%S %z')),
                                    utc=True)
    start, end = df.iloc[0]["datetime"], df.iloc[-1]["datetime"]
    cnt = []
    while start <= end:
        tmp = df[start <= df["datetime"]]
        tmp = tmp[(start + pd.tseries.offsets.Hour(1)) > tmp["datetime"]]
        cnt.append(len(tmp))
        start += pd.tseries.offsets.Hour(1)
    plt.plot(cnt)
    plt.savefig(f"../data/figs/{trace_name}.png")
    plt.clf()


if __name__ == "__main__":
    load_ita_trace("calgary", relative=False)
    exit()
    for tn in ["calgary", "nasa", "usask", "clarknet"]:
        ita_data_analysis(tn)
