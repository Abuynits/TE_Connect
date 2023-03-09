import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.sparse


def permute_tensor(inp: torch.Tensor, permutation, dim=1):
    # restructures tensor; the dimth dimension of tensor has same dim as permutation
    # other dims hold the same size
    return inp.index_select(dim, permutation)


class Reservoir(nn.Module):

    def __init__(self,
                 mode,
                 input_features,
                 hidden_features,
                 num_layers,
                 leaking_rate,
                 spectral_radius,
                 w_ih_scale,
                 density,
                 bias=True,
                 batch_first=False):
        super(Reservoir, self).__init__()

        self._mode = mode
        self._input_features = input_features
        self._hidden_features = hidden_features
        self._num_layers = num_layers
        self._leaking_rate = leaking_rate
        self._spectral_radius = spectral_radius
        self._w_ih_scale = w_ih_scale
        self._density = density
        self._bias = bias
        self._batch_first = batch_first

        self._all_weights = []
        # initialize all weights in each recurrent layer in the ESN
        for layer in range(num_layers):
            # the input and hidden layers will have different sizes
            layer_input_size = self._input_features if layer == 0 else self._hidden_features

            # map from previous recurrent layer to current one (same time step)
            w_ih = nn.Parameter(torch.Tensor(self._hidden_features, self._layer_input_size))
            # map from same current time step to current one (different time step, same recurrent layer)
            w_hh = nn.Parameter(torch.Tensor(self._hidden_features, self._hidden_features))
            # add on bias to recurrent nueral networks - add a constant to a nn
            b_ih = nn.Parameter(torch.Tensor(self._hidden_features))

            layer_params = (w_ih, w_hh, b_ih)
            # track and configure the wights
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            # add on bias param name if used
            if self._bias:
                param_names += ['bias_ih_l{}{}']
            param_names = [x.format(layer,'') for x in param_names]

