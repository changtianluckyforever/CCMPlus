from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.fcn = nn.Linear(1, args.pred_len)

    def forward(self, x_enc, x_mark_enc, batch_y, y_mark_dec=None):
        out = self.fcn(x_enc)
        return out
