try:
    import gpytorch
except ImportError:
    raise ImportError("Please install gpytorch to use this module: 'pip install gpytorch'.")
import copy
import math
from torch import nn
import torch
import torch.nn.functional as F


class InternalMAFE(nn.Module):
    def __init__(self, args, inp_len):
        super(InternalMAFE, self).__init__()
        self.args = args
        self.l_ps = [7, 2]
        self.w_k1 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[0]), int(inp_len / self.l_ps[0])))
        self.w_k2 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[1]), int(inp_len / self.l_ps[1])))
        self.w_v1 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[0]), int(inp_len / self.l_ps[0])))
        self.w_v2 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[1]), int(inp_len / self.l_ps[1])))
        self.h1 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[0]), int(inp_len / self.l_ps[0])))
        self.h2 = nn.Parameter(torch.randn(int(inp_len / self.l_ps[1]), int(inp_len / self.l_ps[1])))
        self.alpha1 = nn.Parameter(torch.randn(1))
        self.alpha2 = nn.Parameter(torch.randn(1))
        self.beta1 = nn.Parameter(torch.randn(1))
        self.beta2 = nn.Parameter(torch.randn(1))
        self.out = []

        self.proj = nn.Linear(len(self.l_ps), 1)
        self.proj_len = nn.Linear(inp_len, args.seq_len)

    def attention(self, x, q, wk, wv):
        att = F.softmax(torch.matmul(q, torch.matmul(x, wk).mT) / math.sqrt(q.shape[-1]), dim=-1)
        out = torch.matmul(x, wv) * att.mT
        return out

    def forward(self, x):
        x_reshape = x.reshape(x.size(0), -1, self.l_ps[0])
        x_out1 = []
        for i in range(x_reshape.size(-1)):
            if i == 0:
                x_out1.append(self.attention(x_reshape[:, :, i], self.h1, self.w_k1, self.w_v1))
            else:
                out = self.attention(x_reshape[:, :, i], self.h1, self.w_k1, self.w_v1)
                gated_out = (
                    torch.tanh(self.alpha1 * x_out1[-1] + self.beta1)
                    * torch.sigmoid(self.alpha2 * x_out1[-1] + self.beta2)
                    + out
                )
                x_out1.append(gated_out)

        x_reshape = x.reshape(x.size(0), -1, self.l_ps[1])
        x_out2 = []
        for i in range(x_reshape.size(-1)):
            if i == 0:
                x_out2.append(self.attention(x_reshape[:, :, i], self.h2, self.w_k2, self.w_v2))
            else:
                out = self.attention(x_reshape[:, :, i], self.h2, self.w_k2, self.w_v2)
                gated_out = (
                    torch.tanh(self.alpha1 * x_out2[-1] + self.beta1)
                    * torch.sigmoid(self.alpha2 * x_out2[-1] + self.beta2)
                    + out
                )
                x_out2.append(gated_out)

        x_out1 = torch.stack(x_out1, dim=-1).view(x.size(0), x.size(1), -1)
        x_out2 = torch.stack(x_out2, dim=-1).view(x.size(0), x.size(1), -1)

        x_out = torch.cat([x_out1, x_out2], dim=-1)
        x_out = self.proj(x_out).squeeze(-1)
        x_out = self.proj_len(x_out1.squeeze(-1))
        return x_out


class ExternalMAFE(nn.Module):
    def __init__(self, args):
        super(ExternalMAFE, self).__init__()
        self.args = args
        self.l_scale = [12, 4, 2, 1]
        self.internal_mafe = nn.ModuleList([InternalMAFE(args, args.seq_len // s) for s in self.l_scale])
        self.fusion_mlp = nn.Linear(args.seq_len, args.seq_len)

        self.ff = nn.Linear(len(self.l_scale), 1)

    def forward(self, x):
        out = []
        for i in range(len(self.l_scale)):
            x_pool = F.avg_pool1d(x, kernel_size=self.l_scale[i])
            out.append(self.internal_mafe[i](x_pool))
        out = torch.stack(out)
        out = self.ff(out.permute(1, 2, 0)).squeeze(-1)
        return out


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.mafe_model = ExternalMAFE(args)
        self.old_x, self.old_y = None, None
        # self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.ff = nn.Linear(args.seq_len, 1)

    def forward_gp(self, x_enc, x_mark_enc, batch_y, y_mark_dec=None):
        dec_out = self.mafe_model(x_enc.squeeze(-1))
        gp_model = ExactGPModel(self.old_x, self.old_y, self.gp_likelihood).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, gp_model)
        if self.old_x is None:
            gp_model.eval()
        else:
            gp_model.train()
        out = gp_model(dec_out)
        pred = self.gp_likelihood(out).mean
        if self.old_x is None:
            mll_loss = torch.tensor(0)
        else:
            mll_loss = mll(out, batch_y).mean()
        self.old_x, self.old_y = dec_out, batch_y
        return pred, mll_loss

    def forward(self, x_enc, x_mark_enc, batch_y, y_mark_dec=None):
        dec_out = self.mafe_model(x_enc.squeeze(-1))
        pred = self.ff(dec_out)
        pred = F.relu(pred)
        return pred, 0
