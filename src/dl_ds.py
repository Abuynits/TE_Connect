from filepaths_constants import *


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
        return x, t, y
