from data_constants import *
from model_constants import *
from DataFormating import *


def print_model(model):
    sum = 0
    print(model)
    for param in model.parameters():
        if param.requires_grad:
            sum += param.numel()
    print("total trainable parameters:", sum)


def createList(l, h):
    return [i for i in range(l, h)]


def print_data_loader(inp_seq, targ_seq, out_seq, num_print):
    for i in range(num_print):
        print(" Input seq:\n", inp_seq[i])
        print(" target seq:\n", targ_seq[i])
        print(" Out seq:\n", out_seq[i])


def check_data_loader(dl_train_item, dl_test_item):
    # print(dl_train_item.shape)
    # print(dl_test_item.shape)
    train_low = 0
    train_high = len(dl_train_item[0])
    test_low = 0
    test_high = len(dl_test_item[0])
    for seq in dl_train_item:
        plt.plot(createList(train_low, train_high), seq.squeeze(), color='blue')
        train_low += 1
        train_high += 1
    plt.title("train plot")
    plt.show()
    for seq in dl_test_item:
        if len(seq) == 1:
            # print(seq.shape)
            # print(createList(test_low,test_high))
            plt.scatter(createList(test_low, test_high), seq)
        else:
            plt.plot(createList(test_low, test_high), seq.squeeze())
        test_low += 1
        test_high += 1
    plt.title("test plot")
    plt.show()


class finance_data_set(Dataset):
    def __init__(self, xDT, targetDT, yDT):
        super().__init__()
        self.x = xDT
        self.y = yDT
        self.t = targetDT
        self.data_len = len(self.t)

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).float()
        y = torch.from_numpy(self.y[i]).float()
        t = torch.from_numpy(self.t[i]).float()
        if torch.cuda.is_available():
            return x.cuda(), t.cuda(), y.cuda()
        return x, t, y


class tft_ds(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = df.reset_index(drop = True)
        # pass in the whole dataframe - with indices, everything.
        self.id_col = get_col_from_inp_type(InputTypes.ID, col_def)
        self.time_col = get_col_from_inp_type(InputTypes.TIME, col_def)
        self.target_col = get_col_from_inp_type(InputTypes.TARGET, col_def)

        self.inp_cols = [
            tup[0]
            for tup in col_def
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        self.col_map = {
            'identifier': [self.id_col],
            'time': [self.time_col],
            'outputs': [self.target_col],
            'inputs': self.inp_cols
        }

        self.lookback = LOOKBACK
        self.num_enc_steps = PREDICT

        self.data_idx = self.get_idx_filtering()
        self.group_size = self.data.groupby([self.id_col]).apply(lambda x: x.shape[0]).mean()
        self.data_index = self.data_idx[self.data_idx.end_rel < self.group_size].reset_index()
        # TODO: need to add id col to dataframe

    def get_index_filtering(self):
        g = self.data.groupby([self.id_col])

        df_index_abs = g[[self.target_col]].transform(lambda x: x.index + self.lookback) \
            .reset_index() \
            .rename(columns={'index': 'init_abs',
                             self.target_col: 'end_abs'})
        df_index_rel_init = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index) \
            .rename(columns={self.target_col: 'init_rel'})
        df_index_rel_end = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index + self.lookback) \
            .rename(columns={self.target_col: 'end_rel'})
        df_total_count = g[[self.target_col]].transform(lambda x: x.shape[0] - self.lookback + 1) \
            .rename(columns={self.target_col: 'group_count'})

        return pd.concat([df_index_abs,
                          df_index_rel_init,
                          df_index_rel_end,
                          self.data[[self.id_col]],
                          df_total_count], axis=1).reset_index(drop=True)
