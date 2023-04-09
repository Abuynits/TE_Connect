from filepaths_constants import *
from model_constants import *
from data_constants import *
import torch.nn as nn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self,
                 n_head,
                 hidden_size,
                 dropout):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        # get dim by taking the floor of hidden size div by number of heads
        self.dim_k = hidden_size // n_head
        self.dim_q = hidden_size // n_head
        self.dim_v = hidden_size // n_head

        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)


class ScaledDotProdAttn(nn.Module):
    def __init__(self,
                 dropout=0):
        super(ScaledDotProdAttn, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # A(Q, K) = Softmax(Q[k^t]/sqrt(d_attn))

        # batch matrix-matrix product of q and k
        # attn = [Q]*[K^T]
        # need to transpose the k matrix by permuting it
        k_t = k.permute(0, 2, 1)
        attn = torch.bmm(q, k_t)

        # calculate dim of attn layer by scaled-dot-product attn
        dim = torch.sqrt(torch.tensor(k.shape[-1])).to(torch.float32)
        attn = attn / dim
        # now done with calculating attention

        # multiply the attn by the value (works with batches)
        out = torch.bmm(attn, v)

        return out, attn
