import gzip
from datetime import datetime

import numpy as np
import pandas as pd

translate = ["translat"]

movie = ["youtube", "video", "film", "cinema", "flick", "motion picture", "blockbuster", "theater", "box office",
         "trailer", "hollywood", "actor", "director", "screenplay", "genre", "plot", "soundtrack", "cinematography",
         "editing", "sequel", "remake", "award-winning movies"]  # + movie list

game = ["game"]  # + game list

food = ["restaurant", "delicious", "pizza", "hamburger", "fried", "noodle"]

image = ["image", "picture", "photo"]

music = ["music", "brand"]

fraud = ["ebay", "pay"]

shopping = ["phone", "clothes", "shop", "amazon"]

digital_map = ["disney", "city", "where", "place", "map"]

urber = ["taxi", "urber", "bus"]

retrieve = ["encyclopedia", "wikipedia", "history", "what", "googl"]


def query_class(keys):
    import matplotlib.pyplot as plt
    data = []
    for i in range(1, 11):
        fid = f"0{i}" if i < 10 else "10"
        with gzip.open(f"AOL-user-ct-collection/user-ct-test-collection-{fid}.txt.gz", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode('utf-8').split("\t")
                data.append({"query": line[1], "time": line[2]})
    df = pd.DataFrame(data)
    print(df.head())
    df["time"] = pd.to_datetime(df["time"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')))
    df = df.sort_values(["time"])
    df.set_index(["time"], inplace=True)
    print(df.head())
    res = pd.DataFrame()
    for key, value in keys.items():
        print(key)
        df_key = df[df["query"].map(lambda x: np.any([k in x for k in value]))]
        df_key = df_key.resample('1H').count().rename({"query": key}, axis="columns")
        if len(res) == 0:
            res = df_key
        else:
            res = df_key.merge(res, on=["time"])
    res.to_csv("query_cls_cnt.csv")
    # plt.figure(figsize=(100, 5))
    # df.resample('1H').count().plot.line()
    # plt.savefig("figs/translate.png")


def plot_cls():
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    df = pd.read_csv("query_cls_cnt.csv")
    print(df.head())
    plt.figure(figsize=(100, 8))
    plt.plot(df["time"], df["food"], label="food")
    plt.plot(df["time"], df["urber"], label="urber")
    plt.plot(df["time"], df["translate"], label="translate")
    plt.xticks(rotation=45)
    x_major_locator = MultipleLocator(12)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig("figs/cls_comp.png")


# query_class({"food": food, "game": game, "translate": translate, "movie": movie, "music": music, "image": image,
#              "fraud": fraud, "shopping": shopping, "digital_map": digital_map, "retrieve": retrieve, "urber": urber})
# plot_cls()
