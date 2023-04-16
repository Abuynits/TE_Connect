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

        # initialize a unique q and k layer for each head
        self.q_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.dim_q, bias=False)
                                       for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.dim_k, bias=False)
                                       for _ in range(self.n_head)])

        # initialize a SINGLE v layer and use the same v for all layers
        self.v_layer = nn.Linear(self.hidden_size, self.dim_v, bias=False)
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])

        self.attention = ScaledDotProdAttn()
        # use for final linear mapping from dim of heads to hidden size used in model
        self.w_h = nn.Linear(self.dim_v, self.hidden_size, bias=False)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        # loop over all heads and gather attn, apply dropout, append to list of all head, attn
        for i in range(self.n_head):
            # pass the q, k, v through each corresponding head in MHA
            v_s = self.v_layers[i](v)
            q_s = self.q_layers[i](q)
            k_s = self.k_layers[i](k)

            # pass each of the heads through the ScaledDotProd
            head, attn = self.attention(q_s, k_s, v_s, mask)
            # apply dropout to head and append attn and head to list of all heads, attns
            head_drop = self.dropout(head)
            heads.append(head_drop)
            attns.append(attn)
        # stack the heads for interpretability
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        # calc the mean attention of all the heads
        outs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # apply linear mapping to the outputs to convert to right dims
        outs = self.w_h(outs)
        # apply dropout to the outputs
        outs = self.dropout(outs)
        return outs, attn


class ScaledDotProdAttn(nn.Module):
    def __init__(self,
                 dropout=0):
        super(ScaledDotProdAttn, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
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
        if mask is not None:
            # create a mask to prevent cheating and looking into the future
            attn = attn.masked_fill(mask == 0, -1e9)
        # apply softmax and dropout after mask
        attn = self.softmax(attn)

        attn = self.dropout(attn)

        # multiply the attn by the value (works with batches)
        out = torch.bmm(attn, v)

        return out, attn
