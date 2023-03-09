import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from .reservoir import Reservoir
from ..utils import washout_tensor


class ESN(nn.Module):
    """
    input_features, # number of features in x
    hidden_features, # number of features in hidden state
    output_features, # number of output features
    num_layers = 1, # # number of recurrent layers
    non_linearity = 'tanh', #linearity used in resovoir
    batch_first = False, # wether have batch first as inp
    leaking_rate = 1, #leaking rate of reservoirs neurons
    spectral_radius = 0.9, # desired spectral radius
    w_ih_scale = 1, # scale factor for first layer input weights
    lambda_reg = 0, # ridge regression shrinkage parameter
    density = 1, # recurrent weight matrix density
    w_io = False, # if true, use trainable input-to-output connections
    readout_training = 'svd', # readouts training algorithm
    options:
    - gd: gradients accumulated during backward pass
        - svd, cholesky, inv: network learn readout's params during forward pass using ridge regression
        ridge regression: model tuninng method used to analyse data suffers from multi collinearity
    output_steps = 'all'
    define how resevours output will be used by ridge regression metho
    all: entire reservoir output matrix is used
    mean: mean of reservoir output matrix along with timesteps is used
    last: only the last timestep of reservoir output matrix is used

    """

    def __init__(self,
                 input_features,
                 hidden_features,
                 output_features,
                 num_layers=1,
                 non_linearity='tanh',
                 batch_first=False,
                 leaking_rate=1,
                 spectral_radius=0.9,
                 w_ih_scale=1,
                 lambda_reg=0,
                 density=1,
                 w_io=False,
                 readout_training='svd',
                 output_steps='all'
                 ):
        super(ESN, self).__init__()

        self._input_features = input_features
        self._hidden_features = hidden_features
        self._output_features = output_features

        if non_linearity == 'tanh':
            self._mode = 'RES_TANH'
        elif non_linearity == 'relu':
            self._mode = 'RES_RELU'
        elif non_linearity == 'id':
            self._mode = 'RES)ID'
        else:
            raise ValueError("Linearity '{}' not found.".format(non_linearity))

        self._batch_first = batch_first
        self._leak_rate = leaking_rate
        self._spectral_radius = spectral_radius

        if type(w_ih_scale) != torch.Tensor:
            self._w_ih_scale = torch.ones(input_features + 1)
            self._w_ih_scale *= w_ih_scale
        else:
            self._w_ih_scale = w_ih_scale

        self._lambda_reg = lambda_reg
        self._density = density
        self._w_io = w_io

        if readout_training in {'gd', 'svd', 'cholesky', 'inv'}:
            self._readout_training = readout_training
        else:
            raise ValueError("Unknown readout training algorithm '{}'".format(readout_training))

        self._reservoir = Reservoir(self._mode,
                                    self._input_features,
                                    self._hidden_features,
                                    self._num_layers,
                                    self._leaking_rate,
                                    self._spectral_radius,
                                    self.w_ih_scale,
                                    self._density,
                                    batch_first=self._batch_first)

        if w_io:
            # layer for reading out the data from the model
            self.readout = nn.Linear(self._input_features + self._input_features * num_layers,
                                     self._output_features)
        else:
            self.readout = nn.Linear(self._hidden_size * num_layers, self._output_features)
        # if dont want to get gradients on the readout layer, disable it
        if readout_training == 'offline':
            self.readout.weight.requires_grad = False

        # Set the steps that are used for sampling from the repo
        if output_steps in {'all', 'mean', 'last'}:
            self.output_steps = output_steps
        else:
            raise ValueError("Unknown task '{}'".format(
                output_steps))

        self.XTX = None
        self.XTy = None
        self.X = None

