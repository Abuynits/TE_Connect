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
                 input_size,
                 hidden_size,
                 num_layers,
                 leaking_rate,
                 spectral_radius,
                 w_ih_scale,
                 density,
                 bias=True,
                 batch_first=False):
        super(Reservoir, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.w_ih_scale = w_ih_scale  # control the scale for the random init of the weights
        self.density = density  # controls how dense the matrix is initialized with
        self.bias = bias
        self.batch_first = batch_first

        self._all_weights = []
        # initialize all weights in each recurrent layer in the ESN
        for layer in range(num_layers):
            # the input and hidden layers will have different sizes
            self.layer_input_size = self.input_size if layer == 0 else self.hidden_size

            # map from previous recurrent layer to current one (same time step)
            w_ih = nn.Parameter(torch.Tensor(self.hidden_size, self.layer_input_size))
            # map from same current time step to current one (different time step, same recurrent layer)
            w_hh = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            # add on bias to recurrent nueral networks - add a constant to a nn
            b_ih = nn.Parameter(torch.Tensor(self.hidden_size))

            layer_params = (w_ih, w_hh, b_ih)
            # track and configure the wights
            param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
            # add on bias param name if used
            if self.bias:
                param_names += ['bias_ih_l{}{}']
            param_names = [x.format(layer, '') for x in param_names]

            # set the param attributes
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            # appends the weights for the parameters
            self.all_weights.append(param_names)

        self.reset_params()

    def reset_params(self):
        # state_dict: used for saving and loading models
        weight_dict = self.state_dict()
        # loop over all weight params and randomize all of them
        for key, value in weight_dict.items():
            # if is the first layer in the mask
            if key == 'weight_ih_l0':
                nn.init.uniform_(value, -1, 1)
                value *= self.w_ih_scale[1:]
            # if hve any other weight layer that's not the first one
            # randomize the weights mapping previous recurrent layer to current layer
            elif re.fullmatch('weight_ih_l[^0]*', key):
                nn.init.uniform_(value, -1, 1)
            # for bias layers
            elif re.fullmatch('bias_ih_l[0-9]*', key):
                nn.init.uniform_(value, -1, 1)
                value *= self.w_ih_scale[0]
            # randomize the weights mapping the previous time step recurrent layer to the current one
            elif re.fullmatch('weight_hh_l[0-9]*', key):
                # create a new tensor
                w_hh = torch.Tensor(self.hidden_size * self.hidden_size)
                w_hh.uniform_(-1, 1)
                # density controls whether have zeros or not
                if self.density < 1:
                    # generate a mask for zero weights
                    zero_weights = torch.randperm(int(self.hidden_size * self.hidden_size))
                    # extracts the zero weights of the matrix
                    zero_weights = zero_weights[:int(self.hidden_size * self.hidden_size * (1 - self.density))]
                    # set the zero weights for the weight matrix
                    w_hh[zero_weights] = 0
                w_hh = w_hh.view(self.hidden_size, self.hidden_size)
                abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
                weight_dict[key] = w_hh * (self.spectral_radius / torch.max(abs_eigs))
        self.__setstate__(weight_dict)
        self.load_state_dict(weight_dict)

    def check_input(self, input, batch_sizes):
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        expected_hidden_size = (self.num_layers, mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        return permute_tensor(hx, permutation)

    def forward(self, input, is_training, hx=None):
        # PackedSequence holds instance of data packed in array
        is_packed = isinstance(input, PackedSequence)
        # extract input, batch size ,indices based on input type
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            # create a blank input hidden layer for hx
            hx = input.new_zeros(self.num_layers, max_batch_size, self.hidden_size, requires_grad=False)

        # check that provided inputs have the correct shape for function
        self.check_forward_args(input, hx, batch_sizes)
        # create a reservoir autograd reservoir

        auto_grad_res = AutogradReservoir(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            train=is_training,
            variable_length=is_packed,
            flat_weight=None,
            leaking_rate=self.leaking_rate,
        )

        output, hidden = auto_grad_res(input, self.all_weights, hx, batch_sizes)

        if is_packed:
            output = PackedSequence(input, self.all_weights, hx, batch_sizes)
        # return the output and hidden layers involved in calculation
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        s += ')'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(Reservoir, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        self._all_weights = []
        for layer in range(num_layers):
            weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}']
            weights = [x.format(layer) for x in weights]
            if self.bias:
                self._all_weights += [weights]
            else:
                self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]


def AutogradReservoir(mode,
                      input_size,
                      hidden_size,
                      num_layers=1,
                      batch_first=False,
                      train=True,
                      batch_sizes=None,
                      variable_length=False,
                      flat_weight=None,
                      leaking_rate=1):
    if mode == 'RES_TANH':
        cell = ResTanhCell
    elif mode == 'RES_RELU':
        cell = ResReLUCell
    elif mode == 'RES_ID':
        cell = ResIdCell
    else:
        raise ValueError("'{}' mode is invalid.".format(mode))

    # determine the layer that will be used in the reservoir
    if variable_length:
        layer = (VariableRecurrent(cell, leaking_rate),)
    else:
        layer = (Recurrent(cell, leaking_rate),)

    # create the reservoir from the stacked RNN
    func = StackedRNN(layer,
                      num_layers,
                      False,
                      train=train)
    def forward(input, weight, hidden, batch_sizes):
        # swap out the input dims to follow the correct dimensions
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)
        # get the predicted output dims
        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and not variable_length:
            output = output.transpose(0, 1)

        return output, nexth

    return forward


# pass through a recurrent cell in a model
def Recurrent(inner, leaking_rate):
    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, leaking_rate, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return hidden, output

    return forward


def VariableRecurrent(inner, leaking_rate):
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], leaking_rate, *weight),)
            else:
                hidden = inner(step_input, hidden, leaking_rate, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


# stacked RNN for iterating through layers
def StackedRNN(inners, num_layers, lstm=False, train=True):
    num_directions = len(inners)
    # number of dims to traverse through
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        print(len(weight))
        print(total_layers)
        assert (len(weight) == total_layers)
        next_hidden = []
        all_layers_output = []

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                # pass through each inner layer and get the hidden layer and output dim
                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)
            all_layers_output.append(input)

        all_layers_output = torch.cat(all_layers_output, -1)
        next_hidden = torch.cat(next_hidden, 0).view(
            total_layers, *next_hidden[0].size())

        return next_hidden, all_layers_output

    return forward


# applies tanh activation function to final recurrent output
def ResTanhCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = torch.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


# applies relu activation function to final recurrent output
def ResReLUCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh))
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy


# doubles the cell in the end
def ResIdCell(input, hidden, leaking_rate, w_ih, w_hh, b_ih=None):
    hy_ = F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh)
    hy = (1 - leaking_rate) * hidden + leaking_rate * hy_
    return hy
