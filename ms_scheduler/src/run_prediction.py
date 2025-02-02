import argparse

import torch
import random
import numpy as np

from prediction.run import Runner

# fix_seed = 2023
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="Workload Prediction")

# basic config
parser.add_argument("--random_seed", type=int, default=0, help="the random seed for experiments")
parser.add_argument(
    "--task_name",
    type=str,
    required=False,
    default="forecast",
    help="task name, options:[long_term_forecast, short_term_forecast]",
)
parser.add_argument("--is_training", type=int, required=False, default=1, help="status")
parser.add_argument("--model_id", type=str, required=False, default="train", help="model id")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="ccformer",
    help="model name, options: [ccformer, Autoformer, Transformer, TimesNet]",
)

# data loader
parser.add_argument(
    "--data", type=str, required=False, default="ALI", help="dataset type, it could be as follows APP_RPC, AZURE, ALI"
)
parser.add_argument("--root_path", type=str, default="../data/processed_data/", help="root path of the data file")
parser.add_argument("--data_path", type=str, default="ali_T_init.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="S",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="location of model checkpoints")

# forecasting task
parser.add_argument("--seq_len", type=int, default=168, help="input sequence length")
parser.add_argument("--label_len", type=int, default=1, help="start token length")
parser.add_argument("--pred_len", type=int, default=1, help="prediction sequence length")
parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")

parser.add_argument("--n_services", type=int, default=128, help="number of services")
parser.add_argument(
    "--low_rank_d",
    type=int,
    default=10,
    help="low rank dimension, the same as the number of K in topK of pearson matrix",
)
parser.add_argument("--low_rank_corr", action="store_true", required=False, help="low_rank_corr")
parser.add_argument("--sparse_corr", action="store_true", required=False, help="sparse_corr")
parser.add_argument("--low_rank_prediction", type=bool, default=True, help="low_rank_prediction")
parser.add_argument("--low_rank_prediction_d", type=int, default=16, help="low rank prediction dimension")

# model define
parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
parser.add_argument("--epsilon", type=float, default=0.1, help="for epsilon greedy in sparse causal matrix")
parser.add_argument(
    "--use_epsilon_greedy_sparse", type=bool, default=False, help="whether to use epsilon greedy sparse"
)
parser.add_argument("--normalize_multiplier", type=bool, default=True, help="whether to normalize multiplier")
parser.add_argument("--use_topk_sparse", type=bool, default=False, help="whether to use topk sparse")
parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=1, help="decoder input size")
parser.add_argument("--c_in", type=int, default=16, help="input channel size")
parser.add_argument("--c_out", type=int, default=1, help="output size")
parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=64, help="dimension of fcn")
parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed", type=str, default="learned", help="time features encoding, options:[timeF, fixed, learned]"
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")

# optimization
parser.add_argument("--num_workers", type=int, default=4, help="data loader num workers")
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
parser.add_argument("--current_epoch", type=int, default=0, help="recording the current epoch for training")
parser.add_argument("--current_iter", type=int, default=0, help="recording the current iteration step for training")
parser.add_argument("--batch_size", type=int, default=2, help="batch size of train input data")
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)
parser.add_argument(
    "--model_training_flag",
    type=bool,
    default=True,
    help="if True, training stage; if False, test or validation stage.",
)
parser.add_argument(
    "--momentum_pearson_flag",
    type=bool,
    default=False,
    help="if True, use the momentum pearson; if False, do not use the momentum pearson.",
)
parser.add_argument(
    "--momentum_pearson_value", type=float, default=0.5, help="momentum value to keep before pearson matrix"
)

# GPU
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

# de-stationary projector params
parser.add_argument(
    "--p_hidden_dims", type=int, nargs="+", default=[128, 128], help="hidden layer dimensions of projector (List)"
)
parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print("Args in experiment:")
print(args)
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = "{}_{}_{}_{}_{}_fq{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.random_seed,
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.freq,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        exp = Runner(args)  # set experiments
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = "{}_{}_{}_{}_fq{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.freq,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )

    exp = Runner(args)  # set experiments
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
