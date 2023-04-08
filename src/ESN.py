import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from reservoir import Reservoir
from model_constants import *
from data_constants import *
from seq2seq_arch import print_model

# https://github.com/stefanonardo/pytorch-esn/blob/master/examples/mnist.py#L22


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

    def __init__(self, input_size=INPUT_DATA_FEATURES,
                 hidden_size=ESN_HIDDEN_FEATURES,
                 output_size=OUTPUT_DATA_FEATURES,
                 num_layers=ESN_NUM_LAYERS,
                 nonlinearity=ESN_NON_LINEARITY,
                 batch_first=ESN_BATCH_FIRST,
                 leaking_rate=ESN_LEAKING_RATE,
                 spectral_radius=ESN_SPECTRAL_RADIUS,
                 w_ih_scale=ESN_W_IH_SCALE,
                 lambda_reg=ESN_LAMBDA_REG,
                 density=ESN_DENSITY,
                 w_io=ESN_W_IO,
                 readout_training=ESN_READ_OUT_TRAINING,
                 output_steps=ESN_OUTPUT_STEPS):

        super(ESN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # configure activation founction between computing reservoir
        if nonlinearity == 'tanh':
            mode = 'RES_TANH'
        elif nonlinearity == 'relu':
            mode = 'RES_RELU'
        elif nonlinearity == 'id':
            mode = 'RES_ID'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        self.batch_first = batch_first
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        if type(w_ih_scale) != torch.Tensor:
            self.w_ih_scale = torch.ones(input_size + 1)
            self.w_ih_scale *= w_ih_scale
        else:
            self.w_ih_scale = w_ih_scale

        self.lambda_reg = lambda_reg
        self.density = density
        self.w_io = w_io
        # if dont want to get gradients on the readout layer, disable it
        if readout_training in {'gd', 'svd', 'cholesky', 'inv'}:
            self.readout_training = readout_training
        else:
            raise ValueError("Unknown readout training algorithm '{}'".format(
                readout_training))

        self.reservoir = Reservoir(mode, input_size, hidden_size, num_layers,
                                   leaking_rate, spectral_radius,
                                   self.w_ih_scale, density,
                                   batch_first=batch_first)

        if w_io:
            self.readout = nn.Linear(input_size + hidden_size * num_layers,
                                     output_size)
        else:
            self.readout = nn.Linear(hidden_size * num_layers, output_size)
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
        print_model(self)

    def forward(self, input, washout, h_0=None, target=None):
        with torch.no_grad():
            is_packed = isinstance(input, PackedSequence)
            # pass the output, hidden sequence and starts through the reservoir.
            output, hidden = self.reservoir(input, h_0)
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
                        X[row:row + seq_lengths[s], 1:] = output[:seq_lengths[s],
                                                          s]
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


def reshape_batch(batch):
    batch = batch.view(batch.size(0), batch.size(1), -1)
    return batch.transpose(0, 1).transpose(0, 2)
