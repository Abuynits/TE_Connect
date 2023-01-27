from constants import *
from dl_ds import *
from saving_reading_data import read_arrs_from_fp,read_dicts_from_fp
from seq2seq_arch import *
from lstm_arch import *
from visualization import *

train_x, train_y, test_x, test_y, valid_x, valid_y = read_arrs_from_fp()

train_ds = finance_data_set(train_x, train_y)
test_ds = finance_data_set(test_x, test_y)
valid_ds = finance_data_set(valid_x, valid_y)

print(train_x.shape)
# create dataloader for train dataset
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"batches in train dl: {len(train_dl)}")
# create dataloader for validation dataset
valid_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"batches in valid dl: {len(valid_dl)}")
# create dataloader for test dataset
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"batches in test dl: {len(test_dl)}")

if CHECK_DL:
    check_data_loader(next(iter(train_dl))[0],next(iter(train_dl))[1])

print_data_loader(next(iter(train_dl))[0],next(iter(train_dl))[1],2)
print_data_loader(next(iter(train_dl))[0],next(iter(train_dl))[1],2)
print(next(iter(train_dl))[0].shape)
print(next(iter(train_dl))[1].shape)
print(next(iter(test_dl))[0].shape)
print(next(iter(test_dl))[1].shape)


train_loss = [] # track training loss
valid_loss = [] # track validation loss

model = seq2seq().to(DEVICE) if (ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ) else lstm().to(DEVICE)

loss_func = nn.MSELoss()
optim = optimizer.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optim, gamma=GAMMA)

num_epochs_run = 0


def train_epoch(dl, epoch):
    print_once = True
    model.train(True)

    epoch_train_loss = 0.
    # loop over training batches
    times_run = 0

    for i, (x, y) in enumerate(dl):
        optim.zero_grad()  # zero gradients
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
            x = x.swapaxes(0, 1)  # want to put data in (seq, batches,num features)
            y = y.swapaxes(0, 1)
        model_out = model.forward(x)

        # squeeze the tensors to account for 1 dim sizes
        model_out = model_out.squeeze()
        y = y.squeeze()

        loss = loss_func(model_out, y)
        epoch_train_loss += loss.item() * x.size(0)

        times_run += x.size(0)

        # compute the loss
        loss.backward()
        # step the optimizer
        optim.step()

    return epoch_train_loss / times_run


def test_epoch(dl, epoch):
    model.train(False)
    epoch_test_loss = 0.
    times_run = 0
    # loop over testing batches
    for i, (x, y) in enumerate(dl):
        if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
            x = x.swapaxes(0, 1).to(DEVICE)  # want to put data in (seq, batches,num features)
            y = y.swapaxes(0, 1).to(DEVICE)
        model_out = model(x)
        # squeeze tensors to account for 1 dim sizes
        model_out = model_out.squeeze()
        y = y.squeeze()

        loss = loss_func(model_out, y)
        epoch_test_loss += loss.item() * x.size(0)
        times_run += x.size(0)

    return epoch_test_loss / times_run


for e in range(EPOCHS):
    avg_train_loss = train_epoch(train_dl, e)
    avg_valid_epoch = test_epoch(valid_dl, e)
    num_epochs_run += 1
    train_loss.append(avg_train_loss)
    valid_loss.append(avg_valid_epoch)
    scheduler.step()
    print(f"epoch {e}: avg train loss: {avg_train_loss} avg val loss: {avg_valid_epoch}")

# Plot the validation and training loss
plot_train_val_loss(train_loss,valid_loss)
plt.show()
