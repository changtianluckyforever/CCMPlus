import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from prediction.timefeatures import time_features


# from sktime.utils import load_data


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


class TSDataset(Dataset):
    def __init__(
        self, root_path, flag="train", size=None, data_path="", scale=True, timeenc=0, freq="h", seasonal_patterns=None
    ):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 整个函数都是load同一个数据集合，然后根据不同的set_type来划分训练集、验证集和测试集
        # 如果scale的话，是根据训练集合的统计特征对整个数据集合进行scale
        df_raw["time"] = pd.to_datetime(df_raw.time)
        df_raw.set_index("time", inplace=True)
        # print(df_raw.shape) # (977, 128)
        """df_raw.columns: ['time', ...(other features), target feature]"""
        df_data = df_raw.resample(self.freq).sum().interpolate()
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # df_data = df_raw.resample('3H').mean().interpolate()
        # self.scale shoud be False, to consider magnitude problem.
        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_data.reset_index()[["time"]][border1:border2]
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.time.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.time.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.time.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.time.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(labels=["time"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(df_stamp["time"].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # univariate time-series to multivariate
        data = np.expand_dims(data, axis=2)
        data_stamp = np.expand_dims(data_stamp, axis=1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.x_samples, self.y_samples, self.x_mark_samples, self.y_mark_samples = self.__get_samples()

    def __get_samples(self):
        x_samples, y_samples, x_mark_samples, y_mark_samples = [], [], [], []
        for idx in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
            s_begin = idx
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            x_samples.append(torch.tensor(self.data_x[s_begin:s_end]))
            y_samples.append(torch.tensor(self.data_y[r_begin:r_end]))
            x_mark_samples.append(torch.tensor(self.data_stamp[s_begin:s_end]))
            y_mark_samples.append(torch.tensor(self.data_stamp[r_begin:r_end]))
        x, x_mark = torch.stack(x_samples).transpose(1, 2), torch.stack(x_mark_samples).transpose(1, 2)
        y, y_mark = torch.stack(y_samples).transpose(1, 2), torch.stack(y_mark_samples).transpose(1, 2)
        return x, y, x_mark, y_mark

    def __getitem__(self, index):
        return self.x_samples[index], self.y_samples[index], self.x_mark_samples[index], self.y_mark_samples[index]

    def __len__(self):
        return len(self.x_samples)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(args, flag):
    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = TSDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, "-- data size: ", len(data_set))
    data_loader = DataLoader(
        data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last
    )
    return data_set, data_loader
