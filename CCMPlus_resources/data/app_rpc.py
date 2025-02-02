import copy

import numpy as np
import pandas as pd


def agg_rpc():
    df = pd.read_csv("app_zone_rpc_hour_encrypted.csv")
    print(len(df))
    agg_df = pd.DataFrame(df.groupby(["app_name", "time"])["value"].agg("sum")).reset_index()
    all_agg_df = None
    for gn, g in agg_df.groupby(["app_name"]):
        g.rename(columns={"value": gn}, inplace=True)
        g.drop(columns=["app_name"], inplace=True)
        # print(g.head())
        g["time"] = pd.to_datetime(g["time"], format="%Y/%m/%d %H:%M")
        g.sort_values(by="time", inplace=True)
        g.reset_index(inplace=True, drop=True)
        # g.to_csv(f"app_rpc/{gn}.csv")
        if all_agg_df is None:
            all_agg_df = copy.deepcopy(g)
        else:
            all_agg_df = all_agg_df.merge(copy.deepcopy(g), how="left", on="time")
    print(all_agg_df.head(50))
    print(len(all_agg_df))
    all_agg_df.to_csv("app_rpc/agg_df.csv")


def post_process():
    df_raw = pd.read_csv("processed_data/app_rpc_agg_df.csv", index_col=0)
    df_raw["time"] = pd.to_datetime(df_raw.time)
    df_raw.set_index("time", inplace=True)
    df_data = df_raw.resample('1H').mean().interpolate()
    # df_data = df_raw.dropna(axis=1)
    print(df_data.shape)
    col_sum = pd.DataFrame((df_data != 0).sum(), columns=["value"])
    col_sum = col_sum[col_sum["value"] == df_data.shape[0]]
    df_c = df_data[col_sum.index.to_list()]
    # print(df_c.describe())
    df_c = pd.DataFrame(df_c, dtype=np.int32)
    print(df_c.head())
    df_c.to_csv("processed_data/app_rpc_filter_H.csv")


def plot():
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    df = pd.read_csv("app_rpc/agg_df.csv", index_col=0)
    print(df.head())
    plt.figure(figsize=(100, 12))
    plt.plot(df["time"], df["13b7c974e225b590aab2be15712cc630e8d022edea5e511e3bd9a829be0ff4f1"], label="1")
    plt.plot(df["time"], df["00caea5988b2410d371264eb16bca4082f5fa691722478d5734cb3e0747ae824"], label="2")
    plt.plot(df["time"], df["116d1040ea95ecaf883bf11491d7511134a38bee642b3914d714ab98b713b6b0"], label="3")
    plt.plot(df["time"], df["513d312c329499744c42e120ab33d2d5efc1692c82d0449a067792e7191185a2"], label="4")
    plt.plot(df["time"], df["7440d82a186ba6ebf9e7d63cfd8e3f93e37801a9d4d4af2e43b46fdb59867841"], label="5")
    plt.plot(df["time"], df["bd920397f9d7a49ed9fcb2415e8a3d2f1c6a03d0d92063467da07766d8f36db6"], label="6")
    plt.xticks(rotation=45)
    x_major_locator = MultipleLocator(12)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig("figs/app_6.png")


def auto_correlation():
    pass


# agg_rpc()
# plot()
post_process()
