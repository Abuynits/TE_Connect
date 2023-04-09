from ..utils.dl_ds import print_model
from ..config.model_constants import *


class encoder_lstm(pl.LightningModule):
    def __init__(self, inp_size, hid_size, dropout, layer_count):
        super(encoder_lstm, self).__init__()
        self._input_size = inp_size  # number of input features
        self._hidden_size = hid_size  # number of features in hidden state
        self._layer_count = layer_count  # number of lstm cells
        self._dropout = dropout  # dropout for lstm

        self._lstm = nn.LSTM(input_size=self._input_size,
                             hidden_size=self._hidden_size,
                             num_layers=self._layer_count,
                             dropout=self._dropout)

        self._drop = nn.Dropout(dropout)

    def forward(self, x):
        # input shape: (seq_len, count units in batch, input size)
        if SEQ2SEQ_VERBOSE:
            print("enc: inp shape", x.shape)
        out, (self.hn, self.cn) = self._lstm(x)
        if SEQ2SEQ_VERBOSE:
            # print("enc: out shape",out.shape)
            print("enc: hn shape", self.hn.shape)

        return out.to(DEVICE), (self.hn, self.cn)

    def init_hidden(self, batch_size):
        h_0 = torch.randn(self._layer_count, batch_size, self._hidden_size).requires_grad_().to(DEVICE)
        c_0 = torch.randn(self._layer_count, batch_size, self._hidden_size).requires_grad_().to(DEVICE)
        return (h_0, c_0)


class decoder_lstm(pl.LightningModule):
    def __init__(self, inp_size, out_size, hid_size, layer_count, drop):
        super(decoder_lstm, self).__init__()
        self._inp_feature_size = inp_size  # number of input features
        self._out_feature_size = out_size  # number of output features
        self._hid_size = hid_size  # size of lstm cells
        self._layer_count = layer_count  # number of hidden lstm cells
        self._dropout = drop  # dropout

        self._lstm = nn.LSTM(input_size=self._inp_feature_size,
                             hidden_size=self._hid_size,
                             num_layers=self._layer_count,
                             dropout=self._dropout)

        self._l_in = nn.Linear(self._hid_size, self._inp_feature_size)
        self._l_out = nn.Linear(self._hid_size, self._out_feature_size)

    def forward(self, x, enc_hidden_states):
        # x: 2d - the last input time step so (batches,input_features)
        # enc_hidden_states: the last hidden decoder time step
        if SEQ2SEQ_VERBOSE:
            print("dec: inp shape", x.shape)
        # lstm_out, (hn, cn) = self._lstm(x,enc_hidden_states)
        lstm_out, (self.hn, self.cn) = self._lstm(x.unsqueeze(0), enc_hidden_states)
        if SEQ2SEQ_VERBOSE:
            print("dec: lstm out shape", lstm_out.shape)
            print("dec: hn shape", self.hn[0].shape)

        output = self._l_in(lstm_out.squeeze(0))  # squeeze becasue added dimenion to x
        final_output = self._l_out(lstm_out.squeeze(0))

        if SEQ2SEQ_VERBOSE:
            print("dec: output shape", output.shape)
            print("dec: final output shape", final_output.shape)
            print("dec: final hn shape", self.hn.shape)
        return final_output, output, (self.hn, self.cn)


class seq2seq(pl.LightningModule):
    def __init__(self, inp_size=INPUT_DATA_FEATURES, out_size=OUTPUT_DATA_FEATURES,
                 hid_size=SEQ2SEQ_HIDDEN_SIZE, layer_count=SEQ2SEQ_LAYER_COUNT,
                 dec_dropout=SEQ2SEQ_DECODER_DROPOUT, enc_dropout=SEQ2SEQ_ENCODER_DROPOUT):
        super(seq2seq, self).__init__()
        self._inp_size = inp_size
        self._hid_size = hid_size
        self._layer_count = layer_count
        self._enc_dropout = enc_dropout
        self._dec_dropout = dec_dropout
        self._out_size = out_size

        self._enc = encoder_lstm(inp_size=self._inp_size, hid_size=self._hid_size, dropout=self._enc_dropout,
                                 layer_count=self._layer_count)
        self._dec = decoder_lstm(inp_size=self._inp_size, out_size=self._out_size, hid_size=self._hid_size,
                                 drop=self._dec_dropout, layer_count=self._layer_count)

        print_model(self)
        self.model_name = "seq2seq"

    # self attention
    # AI coffee break

    def forward(self, inp, target):
        outputs = torch.zeros(PREDICT, inp.size(1), self._out_size)
        enc_hidden = self._enc.init_hidden(inp.size(1))

        if SEQ2SEQ_VERBOSE:
            print("============")
            print("seq2seq: input", inp.shape)
            print("seq2seq: outputs", outputs.shape)
            print("seq2seq: hn", enc_hidden[0].shape)

        enc_out, enc_hidden = self._enc(inp)

        if SEQ2SEQ_VERBOSE:
            print("seq2seq: enc_out", enc_out.shape)
            print("seq2seq: enc_hidden", enc_hidden[0].shape)

        dec_inp = inp[-1, :, :]  # get the last element
        dec_hidden = enc_hidden  # enc hidden is the last hidden state of the elements

        if SEQ2SEQ_VERBOSE:
            print("seq2seq: dec_inp", dec_inp.shape)
            print("seq2seq: dec_hidden", dec_hidden[0].shape)

        for p in range(PREDICT):
            final_dec_out, dec_out, dec_hidden = self._dec(dec_inp, dec_hidden)
            if SEQ2SEQ_VERBOSE:
                print(" seq2seq: final_dec_out", final_dec_out.shape)
                print(" seq2seq: dec_out", dec_out.shape)
                print(" seq2seq: dec_hidden hn", dec_hidden[0].shape)
            outputs[p] = final_dec_out
            if SEQ2SEQ_TRAIN_TYPE == SEQ2SEQ_TRAIN_OPTIONS.TEACHER_FORCING:
                if SEQ2SEQ_VERBOSE:
                    print(target.shape)
                    print(dec_out.shape)
                dec_inp = target[:, p, :]  # target includes the previous lookback,
                if SEQ2SEQ_VERBOSE:
                    print(dec_inp.shape)
            elif SEQ2SEQ_TRAIN_TYPE == SEQ2SEQ_TRAIN_OPTIONS.MIXED_TEACHER_FORCING:
                if SEQ2SEQ_MIXED_TEACHER_FORCING_RATIO < random.random():
                    dec_inp = target[:, p, :]  # target includes the previous lookback,
                else:
                    dec_inp = dec_out
            elif SEQ2SEQ_TRAIN_TYPE == SEQ2SEQ_TRAIN_OPTIONS.GENERAL:
                dec_inp = dec_out
        if SEQ2SEQ_VERBOSE:
            print("seq2seq: outputs", outputs.shape)

        return outputs.to(DEVICE)

    def predict_seq(self, x):
        x = x.unsqueeze(1)  # add in dim of 1
        (hn, cn) = self._enc.init_hidden(1)
        enc_out, (hn, cn) = self._enc(x)

        outputs = torch.zeros(PREDICT, OUTPUT_DATA_FEATURES)
        dec_inp = x[-1, :, :]  # get the last element
        dec_hidden = (hn, cn)  # enc hidden is the last hidden state of the elements

        for p in range(PREDICT):
            final_dec_out, dec_out, dec_hidden = self._dec(dec_inp, dec_hidden)
            if SEQ2SEQ_VERBOSE:
                print(" seq2seq: final_dec_out", final_dec_out.shape)
                print(" seq2seq: dec_out", dec_out.shape)
                print(" seq2seq: dec_hidden hn", dec_hidden[0].shape)
            outputs[p] = final_dec_out
            dec_inp = dec_out
        if SEQ2SEQ_VERBOSE:
            print("seq2seq: outputs", outputs.shape)
        return outputs
