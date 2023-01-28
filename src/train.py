import mlflow

from dl_ds import *
from saving_reading_data import *
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
    check_data_loader(next(iter(train_dl))[0], next(iter(train_dl))[1])

print_data_loader(next(iter(train_dl))[0], next(iter(train_dl))[1], 2)
print_data_loader(next(iter(train_dl))[0], next(iter(train_dl))[1], 2)
print(next(iter(train_dl))[0].shape)
print(next(iter(train_dl))[1].shape)
print(next(iter(test_dl))[0].shape)
print(next(iter(test_dl))[1].shape)

train_loss = []  # track training loss
valid_loss = []  # track validation loss

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

run_ml_flow = EXPERIMENT_SOURCE
if run_ml_flow:
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.start_run()
    except:
        print("mlflow not authenticated, running through git")
        run_ml_flow = False


if run_ml_flow == RUN_TYPE.MLFLOW_RUN:
    run = mlflow.active_run()
    run_id = run.info.run_id
# log training parameters
start_time = time.time()

for e in range(EPOCHS):
    avg_train_loss = train_epoch(train_dl, e)
    avg_valid_loss = test_epoch(valid_dl, e)
    num_epochs_run += 1
    train_loss.append(avg_train_loss)
    valid_loss.append(avg_valid_loss)
    scheduler.step()
    if run_ml_flow == RUN_TYPE.MLFLOW_RUN:
        mlflow.log_metric("avg train loss", avg_train_loss, step=e)
        mlflow.log_metric("avg validation loss", avg_valid_loss, step=e)
    print(f"epoch {e}: avg train loss: {avg_train_loss} avg val loss: {avg_valid_loss}")

train_time = time.time() - start_time

best_val_loss = min(train_loss)
last_val_loss = valid_loss[-1]
best_train_loss = min(valid_loss)
last_train_loss = train_loss[-1]

train_run_params = {
    "best validation loss": best_val_loss,
    "last validation loss": last_val_loss,
    "best train loss": best_train_loss,
    "last train loss": last_train_loss,
    "total train time": train_time
}

if run_ml_flow == RUN_TYPE.MLFLOW_RUN:
    print("Done Training!!!")
    print("saving param to MLflow...")
    mlflow.log_params(train_run_params)
    mlflow.log_params(DATA_PREP_DICT)
    mlflow.log_params(MODEL_PARAM_DICT)
    print("saving model to MLflow...")
    model_uri = mlflow.get_registry_uri()
    mlflow.pytorch.log_model(model, MODEL_SAVE_PATH)
    mlflow.end_run()
    #mlflow.pytorch.save_model(model, MODEL_SAVE_PATH)
    # mlflow.register_model(f'runs:/{run_id}/{MODEL_CHOICE}', model)
else:
    print("saving to git!!!!")
    print("Save model....")
    save_model(model)
    print("save data params...")
    save_json(DATA_PREP_DICT, DATA_PARAM_FILE_PATH)
    print("Save run params....")
    save_json(MODEL_PARAM_DICT, MODEL_PARAM_FILE_PATH)
    print("saving train run params...", )
    save_json(train_run_params, MODEL_TRAIN_METRICS_FILE_PATH)
print("done!!!")
# Plot the validation and training loss

