import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate


def tabulate_print(data_frame, rows=5, headers="keys", tail=False):
    print(f"DataFrame shape: {data_frame.shape}")
    if tail:
        print(tabulate(data_frame.tail(rows), headers=headers, tablefmt='psql'))
    else:
        print(tabulate(data_frame.head(rows), headers=headers, tablefmt='psql'))


OUT_DIR = './ali_trace_processed/'


def postprocess():
    df = pd.read_csv('dfas.csv', index_col=0)
    print(df.shape)
    print(all(df["status_x"] == 'Terminated'))
    print(all(df["status_j"] == 'Terminated'))
    print(all(df["status_i"] == 'Terminated'))
    fields = ['job_name', 'task_name', 'inst_num_x', 'gpu_type_x', 'inst_id_x',
              'runtime_i', 'wait_time', 'gpu_type_spec_x', 'group_x', 'workload_x', 'inst_name',
              'worker_name', 'plan_cpu', 'plan_mem', 'plan_gpu', 'cpu_usage',
              'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'read_count',
              'write_count']
    df[fields].to_csv("ali_evaluator_trace.csv")
    # df.drop(columns=["inst_id", "start_date", "duration_min", "status", "status_j", "status_i"], inplace=True)
    print(df.columns.tolist())
    print(set(df["workload_x"]))
    # df.to_csv(OUT_DIR + "ali_task_trace.csv")


def sample(k=10000, part=5):
    df = pd.read_csv('ali_evaluator_trace.csv', index_col=0)
    print(len(df))
    df_g = df.groupby(["job_name"])
    print(len(df_g))
    cnt = 0
    jl = []
    for name, group in df_g:
        if len(group["task_name"].unique()) > 1 or group.iloc[0]["task_name"] != "evaluator":
            continue
        jl.append(group)
        cnt += 1
        if cnt >= k * part:
            break
    df_k = pd.concat(jl).reset_index(drop=True)
    # df_k.drop(columns=["start_time", "end_time", "runtime_i", "start_time_i", "end_time_i", "end_time_j", "wait_time"],
    #           inplace=True)
    # df_k.rename(columns={"start_time_j": "submit_time", "runtime": "trace_time"}, inplace=True)
    df_k["plan_gpu"] /= 100.0
    df_k["plan_cpu"] /= 100.0
    print("more than 1 GPU card, ", any(df_k["plan_gpu"] > 1.0), sum(df_k["plan_gpu"] > 1.0))
    """
            count    50000.000000
            mean         1.897865
            std          4.691435
            min          0.010000
            25%          0.250000
            50%          1.000000
            75%          1.000000
            max         98.000000
    """
    print(df_k["plan_gpu"].describe())
    # min_start = df_k["submit_time"].min()
    # df_k["submit_time"] -= min_start
    G = pow(2, 30)
    df_k["read"] /= G
    df_k["write"] /= G
    # df_k = df_k.sort_values(by=['submit_time'])
    print(df_k.columns.tolist())
    tabulate_print(df_k)
    print(len(df_k), len(df_k.groupby(["job_name"])))
    for pi in range(part):
        df_k[pi * k:(pi + 1) * k].to_csv(f'ali_evaluator_trace_10k_p{pi}.csv')


def extract_time_feat():
    for i in range(5):
        df = pd.read_csv(f"./ali_trace_processed/ali_trace_1task_10k_p{i}.csv")
        df["submit_date"] = df["submit_time"].apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
        df['day'] = df['submit_date'].apply(lambda x: x.day)
        df['dayofweek'] = df['submit_date'].apply(lambda d: d.dayofweek)

        # 第五步： 提取与时刻有关的特征
        df['hour'] = df['submit_date'].apply(lambda d: d.hour)

        # 第六步：使用pd.cut将hour的数据进行切分，分成几个过程
        cut_hour = [-1, 5, 11, 16, 21, 23]
        cut_labels = ['last night', 'morning', 'afternoon', 'evening', 'Night']
        df['Hour_cut'] = pd.cut(df['hour'], bins=cut_hour, labels=cut_labels)
        print(df['Hour_cut'].head())
        df = df.reset_index(drop=True)
        tabulate_print(df.head())
        df.to_csv(f"./ali_trace_processed/ali_trace_1task_10k_p{i}.csv")


if __name__ == "__main__":
    # postprocess()
    sample()
    # extract_time_feat()
