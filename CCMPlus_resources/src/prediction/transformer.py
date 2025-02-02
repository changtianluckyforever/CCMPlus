from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from prediction.layers.Transformer_EncDec import Encoder, EncoderLayer
from prediction.layers.self_att_family import FullAttention, AttentionLayer
from prediction.layers.embed import DataEmbedding


class TransformerEncoder(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(TransformerEncoder, self).__init__()
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc, x_mark_enc):
        # the shape of x_enc is [B, N, Lx, H],
        # in this project H is 1
        # the shape of x_mark_enc is [B, 1, Lx, 4]
        # Lx = L, H  = P = 1
        B, N, L, P = x_enc.shape
        x_mark_enc = x_mark_enc.expand(-1, N, -1, -1)
        # the shape of x_mark_enc is [B, N, Lx, 4]
        x_enc = rearrange(x_enc, "b n l h -> (b n) l h")
        x_mark_enc = rearrange(x_mark_enc, "b n l h -> (b n) l h")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # the shape of enc_out is [B * N, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = torch.mean(enc_out, dim=-2)
        enc_out = enc_out.reshape(B, N, -1)
        return enc_out
