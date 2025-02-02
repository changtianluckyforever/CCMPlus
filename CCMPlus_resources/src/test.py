from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.data_util import read_hourly_trace, load_hourly_trace


def test_next_batch():
    ms_in_hour = pd.date_range("2020-08-03 15:00:00", "2020-08-05 15:00:00", freq="ms", inclusive='left')
    ms_in_hour = pd.Series(np.ones(len(ms_in_hour), dtype=np.int8), name="value", index=ms_in_hour)
    time_p = ms_in_hour.sample(5653910)
    time_p = time_p.resample("1T").agg("sum")
    print(time_p.head())
    ds_list = [5470, 5383, 5416, 5332, 5437, 6948, 7089, 6915, 7056, 7215]
    interval = 30 * 60 * 1000
    bsz = 256
    timeout = 10
    ds = ds_list[0]
    batch_num = int(ds / bsz)
    batch_num += batch_num * timeout


def test_dstack():
    a = np.random.random((4, 10))
    b = np.random.normal(0.5, 0.1, (4, 10)).clip(0.4, 0.6)
    print(b)
    print(a, a.shape)
    t1 = a * b
    t2 = a * (1 - b)
    print(t1, t1.shape)
    print(t2, t2.shape)
    t = np.dstack((t1, t2)).reshape((4, 10 * 2))
    print(t, t.shape)


def test_causal_ccm():
    from causal_ccm.causal_ccm import ccm
    df_raw = read_hourly_trace()
    sampled = load_hourly_trace(df_raw, num_models=2, days=60, init_days=7, interval="1H")[0]
    print(sampled.shape)
    X, Y = sampled[0], sampled[1]

    plt.plot(X, linewidth=1.25, label='X')
    plt.plot(Y, c='r', linewidth=1.25, label='Y')
    plt.xlabel('timestep', size=15)
    plt.ylabel('value of X or Y', size=15)
    plt.legend()
    plt.savefig("../exps/XY_ts.png")
    plt.clf()

    tau = 1  # time lag
    E = 2  # shadow manifold embedding dimensions
    L = len(X)  # length of time period to consider

    ccm1 = ccm(X, Y, tau, E, L)
    # causality X -> Y
    # returns: (correlation ("strength" of causality), p-value(significance))
    print(ccm1.causality())

    # visualize sample cross mapping
    # ccm1.visualize_cross_mapping()

    # Check correlation plot
    # ccm1.plot_ccm_correls()

    # checking convergence
    # Looking at "convergence"
    L_range = range(50, 1440, 20)  # L values to test
    tau = 2
    E = 2

    Xhat_My, Yhat_Mx = [], []  # correlation list
    for L in tqdm(L_range):
        ccm_XY = ccm(X, Y, tau, E, L)  # define new ccm object # Testing for X -> Y
        ccm_YX = ccm(Y, X, tau, E, L)  # define new ccm object # Testing for Y -> X
        Xhat_My.append(ccm_XY.causality()[0])
        Yhat_Mx.append(ccm_YX.causality()[0])

    # plot convergence as L->inf. Convergence is necessary to conclude causality
    # plt.figure(figsize=(5, 5))
    plt.plot(L_range, Xhat_My, label='$\hat{X}(t)|M_y$')
    plt.plot(L_range, Yhat_Mx, c='r', label='$\hat{Y}(t)|M_x$')
    plt.xlabel('L')
    plt.ylabel('correl')
    plt.legend()
    plt.savefig("../exps/XY_convergence.png")
    plt.clf()


def test_skccm():
    import skccm as ccm
    df_raw = read_hourly_trace()
    sampled = load_hourly_trace(df_raw, num_models=2, days=30, interval="1H")
    print(sampled.shape)
    x1, x2 = sampled[0], sampled[1]
    lag = 1
    embed = 2
    e1 = ccm.Embed(x1)
    e2 = ccm.Embed(x2)
    X1 = e1.embed_vectors_1d(lag, embed)
    X2 = e2.embed_vectors_1d(lag, embed)


def test_fft():
    df_raw = read_hourly_trace()
    sampled = load_hourly_trace(df_raw, num_models=512, days=0, init_days=7, interval="1H")[0]
    print(sampled.shape)
    mean_0 = np.mean(sampled)
    mean_1 = np.mean(sampled[1:] - sampled[:-1])
    t = 2 * np.sqrt(3 * abs(mean_0) / (abs(mean_1) + 1e-6))
    print(t)

    def show(ori_func, ft, idx, sampling_period=10.):
        n = len(ori_func)
        interval = sampling_period / n
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(0, n), ori_func, 'black')
        plt.xlabel('Time'), plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        frequency = np.arange(n / 2)
        nfft = abs(ft[range(int(n / 2))] / n)
        print(np.argsort(nfft)[::-1][:5])
        plt.plot(frequency, nfft, 'red')
        plt.xlabel('Freq (Hz)'), plt.ylabel('Amp. Spectrum')
        plt.savefig(f"../exps/fft_{idx}.png")
        plt.clf()

    lag = []
    for i in range(len(sampled)):
        y = np.fft.fft(sampled[i])
        nfft = abs(y[range(int(len(y) / 2))] / len(y))
        lag.extend(np.argsort(nfft)[::-1][1:4])
        # show(sampled[i], y, i)

    # Counter({1: 729, 127: 656, 3: 415, 2: 374, 4: 245, 5: 162, 7: 98, 11: 93, 6: 74, 18: 61, 8: 28, 254: 26, 36: 12, 54: 12, 381: 9, 38: 6})
    print(Counter(lag))


class DiaConvs(nn.Module):
    def __init__(self):
        super(DiaConvs, self).__init__()
        dilations = [1, 2, 3, 4, 10]
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3,
                       bias=False, dilation=d) for d in dilations])

    def forward(self, x):
        for conv in self.convs:
            out = conv(x)
            print(out.shape)


def test_dia_convs():
    x = torch.randn(8, 10, 128, 5)
    x_fft = torch.fft.fft(x.mT, dim=-1)
    print(x_fft.shape)
    exit()
    res = x_fft * torch.conj(x_fft)
    print(res)
    exit()
    convs = DiaConvs()
    inx = rearrange(x, "b m l h -> (b m) h l")
    print("inx ", inx.shape)
    convs(inx)


def test_main_embed():
    from modules.ccformer import MultiManifoldEmbedding
    from modules.ccformer import Encoder
    from modules.ccformer import EncoderLayer
    from modules.cclayers import TemporalEmbedding, SeasonalLayerNorm
    m = Encoder(
        embed=TemporalEmbedding(16),
        attn_layers=[
            EncoderLayer(
                MultiManifoldEmbedding(5, 16, taus=np.array([3, 4, 5])),
                16,
                16,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(3)
        ],
        norm_layer=SeasonalLayerNorm(16)
    )
    x = torch.randn(4, 10, 128, 5)
    m(x[..., 0], x[..., 1:])


for _ in range(10):
    test_fft()
    print()
