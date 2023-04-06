import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from reservoir import *
from data_constants import *
from model_constants import *


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
                 input_features=INPUT_DATA_FEATURES,
                 hidden_features=ESN_HIDDEN_FEATURES,
                 output_features=OUTPUT_DATA_FEATURES,
                 num_layers=ESN_NUM_LAYERS,
                 non_linearity=ESN_NON_LINEARITY,
                 batch_first=ESN_BATCH_FIRST,
                 leaking_rate=ESN_LEAKING_RATE,
                 spectral_radius=ESN_SPECTRAL_RADIUS,
                 w_ih_scale=ESN_W_IH_SCALE,
                 lambda_reg=ESN_LAMBDA_REG,
                 density=ESN_DENSITY,
                 w_io=ESN_W_IO,
                 readout_training=ESN_READ_OUT_TRAINING,
                 output_steps=ESN_OUTPUT_STEPS
                 ):
        super(ESN, self).__init__()

        self._input_size = input_features
        self._hidden_size = hidden_features
        self._num_layers = num_layers
        self._output_features = output_features

        if non_linearity == 'tanh':
            self._mode = 'RES_TANH'
        elif non_linearity == 'relu':
            self._mode = 'RES_RELU'
        elif non_linearity == 'id':
            self._mode = 'RES_ID'
        else:
            raise ValueError("Linearity '{}' not found.".format(non_linearity))

        self._batch_first = batch_first
        self._leaking_rate = leaking_rate
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
                                    self._input_size,
                                    self._hidden_size,
                                    self._num_layers,
                                    self._leaking_rate,
                                    self._spectral_radius,
                                    self._w_ih_scale,
                                    self._density,
                                    batch_first=self._batch_first)

        if w_io:
            # layer for reading out the data from the model
            self.readout = nn.Linear(self._input_size + self._input_size * num_layers,
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

    def forward(self, input, washout, h_0=None, target=None):
        with torch.no_grad():
            is_packed = isinstance(input, PackedSequence)
            # pass the output, hidden sequence and starts through the reservoir.
            output, hidden = self._reservoir(input, h_0)
            if is_packed:
                # if have sequences of unequal length, pad the output to fit it
                output, seq_lengths = pad_packed_sequence(output,
                                                          batch_first=self.batch_first)
            else:
                if self.batch_first:
                    # perform in parallel: have the batch (number of samples) times the batch eln
                    seq_lengths = output.size(0) * [output.size(1)]
                else:
                    # batch has been transposed: have sequences of size(1) and batch is added on
                    seq_lengths = output.size(1) * [output.size(0)]

            # want to have the first dim be the number of examples in a sequence
            if self.batch_first:
                output = output.transpose(0, 1)

            output, seq_lengths = washout_tensor(output, washout, seq_lengths)

            if self.w_io:
                if is_packed:
                    input, input_lengths = pad_packed_sequence(input,
                                                               batch_first=self.batch_first)
                else:
                    input_lengths = [input.size(0)] * input.size(1)

                if self.batch_first:
                    input = input.transpose(0, 1)

                input, _ = washout_tensor(input, washout, input_lengths)
                output = torch.cat([input, output], -1)

            if self.readout_training == 'gd' or target is None:
                with torch.enable_grad():
                    output = self.readout(output)

                    if is_packed:
                        for i in range(output.size(1)):
                            if seq_lengths[i] < output.size(0):
                                output[seq_lengths[i]:, i] = 0

                    if self.batch_first:
                        output = output.transpose(0, 1)

                    # Uncomment if you want packed output.
                    # if is_packed:
                    #     output = pack_padded_sequence(output, seq_lengths,
                    #                                   batch_first=self.batch_first)

                    return output, hidden

            else:
                batch_size = output.size(1)

                X = torch.ones(target.size(0), 1 + output.size(2), device=target.device)
                row = 0
                for s in range(batch_size):
                    if self.output_steps == 'all':
                        X[row:row + seq_lengths[s], 1:] = output[:seq_lengths[s], s]
                        row += seq_lengths[s]
                    elif self.output_steps == 'mean':
                        X[row, 1:] = torch.mean(output[:seq_lengths[s], s], 0)
                        row += 1
                    elif self.output_steps == 'last':
                        X[row, 1:] = output[seq_lengths[s] - 1, s]
                        row += 1

                if self.readout_training == 'cholesky':
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                elif self.readout_training == 'svd':
                    # Scikit-Learn SVD solver for ridge regression.
                    U, s, V = torch.svd(X)
                    idx = s > 1e-15  # same default value as scipy.linalg.pinv
                    s_nnz = s[idx][:, None]
                    UTy = torch.mm(U.t(), target)
                    d = torch.zeros(s.size(0), 1, device=X.device)
                    d[idx] = s_nnz / (s_nnz ** 2 + self.lambda_reg)
                    d_UT_y = d * UTy
                    W = torch.mm(V, d_UT_y).t()

                    self.readout.bias = nn.Parameter(W[:, 0])
                    self.readout.weight = nn.Parameter(W[:, 1:])
                elif self.readout_training == 'inv':
                    self.X = X
                    if self.XTX is None:
                        self.XTX = torch.mm(X.t(), X)
                        self.XTy = torch.mm(X.t(), target)
                    else:
                        self.XTX += torch.mm(X.t(), X)
                        self.XTy += torch.mm(X.t(), target)

                return None, None


def washout_tensor(tensor, washout, seq_lengths, bidirectional=False, batch_first=False):
    # create a tensor such that the batch is the second dim
    tensor = tensor.transpose(0, 1) if batch_first else tensor.clone()
    # handle different types of inputs for the sequence
    if type(seq_lengths) == list:
        seq_lengths = seq_lengths.copy()
    if type(seq_lengths) == torch.Tensor:
        seq_lengths = seq_lengths.clone()
    for b in range(tensor.size(1)):
        if washout[b] > 0:
            tmp = tensor[washout[b]:seq_lengths[b], b].clone()
            tensor[:seq_lengths[b] - washout[b], b] = tmp
            tensor[seq_lengths[b] - washout[b]:, b] = 0
            seq_lengths[b] -= washout[b]

            if bidirectional:
                tensor[seq_lengths[b] - washout[b]:, b] = 0
                seq_lengths[b] -= washout[b]
    # account for output of the sequence
    if type(seq_lengths) == list:
        max_len = max(seq_lengths)
    else:
        max_len = max(seq_lengths).item()

    return tensor[:max_len], seq_lengths
