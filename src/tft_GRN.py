from filepaths_constants import *
from model_constants import *
from data_constants import *
import torch.nn as nn


class GLU(nn.Module):
    # activation is Sigmoid by default
    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout):
        super(GLU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        # configure dropout
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

        # crete weights for GLU function (input_features x output_features)

        self.w4 = nn.Linear(self.input_size, self.hidden_size)  # w_4 * input + b_4
        self.w5 = nn.Linear(self.input_size, self.hidden_size)  # w_5 * input + b_5

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        # all params in the model (w_4, w_4, b_4, b_5)
        for n, param in self.named_parameters():
            if 'bias' not in n:
                nn.init.xavier_normal_(param)  # fill w_i with normal dist values
            elif 'bias' in n:
                nn.init.zeros_(param)  # fill b_i values with 0's

    def forward(self, inp):

        if self.dropout:
            inp = self.dropout(inp)

        out = self.sigmoid(self.w4(inp) * self.w5(inp))

        return out


class GateAddNorm(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout):
        super(GateAddNorm, self).__init__()
        # create GLU layer for processing input
        self.GLU = GLU(input_size, hidden_size, dropout)

        # create LayerNorm for final output Processing
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp, skip):
        out = self.layer_norm(self.GLU(inp) + skip)
        return out


class GRN(nn.Module):
    def __init__(self,
                 hidden_size,
                 input_size=None,
                 output_size=None,
                 dropout=None,
                 context=None,
                 return_gate=False):
        super(GRN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        if input_size is None:
            self.input_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        # handle additional static context if needed
        self.context = context

        # second processing n1 = W-1 * n2 + b1 where n2 in R^hidden_size
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size)

        # first processing of n2 = ELU(w2 * a + w_3 * ctx + b_2) where a, c in R^input_size
        self.w2 = nn.Linear(self.input_size, self.hidden_size)
        if self.context:
            self.w3 = nn.Linear(self.input_size, self.hidden_size)
        # elu activation: x >= 0: x otherwise exp(x) - 1
        self.ELU = nn.ELU()

        if self.output_size:
            # make a go from input dim to output dim (skipping ELU)
            self.skip_linear = nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNorm(self.hidden_size,
                                            self.output_size,
                                            self.dropout)
        else:
            # no output specified: output should be in hidden dim
            self.glu_add_norm = GateAddNorm(self.hidden_size,
                                            self.hidden_size,
                                            self.dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            print(name)
            if ('w2' in name or 'w3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'w1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(param)
                #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def forward(self, inp):
        if self.context:
            inp, ctx = inp  # extract both context and input from the param
            n2 = self.ELU(self.w2(inp) + self.w3(ctx))
        else:
            # ctx vector is 0 -> ignore
            n2 = self.ELU(self.w2(inp))
        # pass through second dense layer
        n1 = self.w1(n2)

        if self.output_size:
            # have the n1 and the skip layer with provided output size
            out = self.glu_add_norm(n1, self.skip_linear(inp))
        else:
            # still have a hidden size: still implement skip over the layer
            out = self.glu_add_norm(n1, inp)
        return out
