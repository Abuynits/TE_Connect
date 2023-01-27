from constants import *
from dl_ds import print_model
class lstm(nn.Module):
    def __init__(self, inp_size=INPUT_DATA_FEATURES, out_size=INPUT_DATA_FEATURES, hidden_size=LSTM_HIDDEN_SIZE,
                 layer_count=LSTM_LAYER_COUNT, seq_len=LOOKBACK, dropout=0):
        super(lstm, self).__init__()
        self.model_name = "lstm"
        self.inp_size = inp_size
        self.out_size = out_size
        self.hid_size = hidden_size
        self.layer_count = layer_count
        self.seq_len = seq_len
        self.drop = dropout

        self.lstm = nn.LSTM(input_size=self.inp_size, hidden_size=self.hid_size, num_layers=self.layer_count,
                            batch_first=True, dropout=self.drop)

        self.fc = nn.Linear(hidden_size, out_size)

        print_model(self)

    def forward(self, inp):
        (hn, cn) = self.init_hidden(inp.size(0))
        if LSTM_VERBOSE:
            print("inp shape:", inp.shape)
            print("hn shape:", hn.shape)
            print("cn shape:", cn.shape)
        # try with (hn,cn)
        lstm_out, hidden = self.lstm(inp, (hn, cn))
        out = self.fc(hn[0]).flatten()
        if LSTM_VERBOSE:
            print("hn[0] shape:", hn[0].shape)
            print("out lstm shape:", out.shape)
        return out

    def init_hidden(self, b_size):
        h_0 = torch.randn(self.layer_count, b_size, self.hid_size).to(
            DEVICE)  # lstm cells, batches, hidden features
        c_0 = torch.randn(self.layer_count, b_size, self.hid_size).to(
            DEVICE)  # lstm cells, batches, hidden features
        return (h_0, c_0)
