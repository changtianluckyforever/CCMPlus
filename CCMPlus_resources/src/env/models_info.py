import random
import numpy as np
import pandas as pd

from utils.data_util import load_ita_trace, load_hourly_trace, read_hourly_trace

MIN_BSZ, MAX_BSZ = 0, 10
NTM = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
TextCNN = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
           "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
BERT = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
        "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
Transformer = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 7, "pal": 2,
               "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                       8: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                       9: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                       10: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1}}}
GPT1 = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 5, "pal": 2,
        "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                6: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                7: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                8: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                9: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                10: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1}}}
GPT2 = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 4, "pal": 2,
        "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                5: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                6: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                7: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                8: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                9: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1},
                10: {"cpu": -1, "gpu": -1, "mem": -1, "lat": -1}}}
GCN = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
GAT = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
GraphSAGE = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
             "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
Inception = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
             "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                     10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
VGG = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
ResNet = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
          "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
RCNN = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
        "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
ViT = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
WideDeep = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
            "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                    10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
DeepFM = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
          "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
GRU4Rec = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
           "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                   10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
SASRec = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
          "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
                  10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
DQN = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}
PPO = {"lau_cost": 2, "timeout": 0.1, "max_bsz": 10, "pal": 2,
       "bsz": {0: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               1: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               2: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               3: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               4: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               5: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               6: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               7: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               8: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               9: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01},
               10: {"cpu": 2, "gpu": 1, "mem": 4, "lat": 0.01}}}

model_list = [NTM, TextCNN, BERT, Transformer, GPT1, GPT2, GCN, GAT, GraphSAGE, Inception, VGG, ResNet, RCNN, ViT,
              WideDeep, DeepFM, GRU4Rec, SASRec, DQN, PPO]


class ModelSelector:
    def __init__(self, k=5):
        self.k = k
        self.models = None
        self.trace_df = read_hourly_trace()

    def sample_models(self):
        #self.models = random.choices(model_list, k=self.k)
        self.models = model_list 
        desired_util = np.random.normal(loc=0.7, scale=0.5, size=self.k).clip(0.5, 0.9)
        for i in range(self.k):
            self.models[i]["desired_util"] = desired_util[i]

    @property
    def mid(self):
        return self.models.index.to_numpy()

    @property
    def gpus(self):
        return np.array([[v["gpu"] for v in m["bsz"].values()] for m in self.models])

    @property
    def cpus(self):
        return np.array([[v["cpu"] for v in m["bsz"].values()] for m in self.models])

    @property
    def mems(self):
        return np.array([[v["mem"] for v in m["bsz"].values()] for m in self.models])

    @property
    def lat(self):
        return np.array([[v["lat"] for v in m["bsz"].values()] for m in self.models])

    @property
    def max_bsz(self):
        return np.array([m["max_bsz"] for m in self.models])

    @property
    def bsz(self):
        # init bsz as half of the max_bsz
        return (self.max_bsz / 4).astype(int)

    @property
    def pal(self):
        return np.array([m["pal"] for m in self.models])

    @property
    def timeout(self):
        return np.array([m["timeout"] for m in self.models])

    @property
    def launch_cost(self):
        return np.array([m["lau_cost"] for m in self.models])

    @property
    def desired_utils(self):
        return np.array([m["desired_util"] for m in self.models])

    def requests_streaming(self):
        req, req_sizes = [], []
        while len(req) < self.k:
            start_day = random.randint(1, 7)
            for i in range(self.k):
                t = load_ita_trace(random.choice(["calgary", "nasa", "clarknet", "usask"]), weekday=start_day)
                if t is None:
                    req, req_sizes = [], []
                    break
                req.append(t)
                req_sizes.append(len(t))
        return req, req_sizes

    def requests(self, trace_days, init_days=7, sim_poisson=False):
        if sim_poisson:
            return np.random.poisson(5e5, size=(self.k, trace_days * 48))
        traces = load_hourly_trace(self.trace_df, self.k, trace_days, init_days)
        return traces

    def models_info(self):
        return self.gpus, self.cpus, self.mems, self.max_bsz, self.bsz, self.pal, self.lat, self.timeout, \
            self.launch_cost
