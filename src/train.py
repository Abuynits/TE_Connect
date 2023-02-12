import mlflow

from dl_ds import *
from saving_reading_data import *
from seq2seq_arch import *
from lstm_arch import *
from time_transformer import *
from eval import *
from visualization import *

print("reading data from files..")
train_x, train_y, train_tg, valid_x, valid_y, valid_tg = read_train_arrs_from_fp()
print("creating datasets...")
train_ds = finance_data_set(train_x, train_tg, train_y)

valid_ds = finance_data_set(valid_x, valid_tg, valid_y)

# create dataloader for train dataset
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"batches in train dl: {len(train_dl)}")
# create dataloader for validation dataset
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"batches in valid dl: {len(valid_dl)}")

if CHECK_DL:
    check_data_loader(next(iter(train_dl))[0], next(iter(train_dl))[1])
dl_unit = next(iter(train_dl))
print_data_loader(dl_unit[0], dl_unit[1], dl_unit[2], 2)

print(next(iter(train_dl))[0].shape)
print(next(iter(train_dl))[1].shape)
print(next(iter(train_dl))[2].shape)

train_loss = []  # track training loss
valid_loss = []  # track validation loss

if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
    model = seq2seq().to(DEVICE)
elif ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:
    model = lstm().to(DEVICE)
elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
    model = time_transformer().to(DEVICE)
else:
    raise Exception("bad model selected!")
loss_func = nn.MSELoss()
optim = optimizer.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ExponentialLR(optim, gamma=GAMMA)

num_epochs_run = 0


def get_model_pred(x, target, y):
    if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
        x = x.swapaxes(0, 1)  # want to put data in (seq, batches,num features)
        y = y.swapaxes(0, 1)
        model_out = model.forward(x)
    elif ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:
        model_out = model.forward(x)
    elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
        # input mask: [prediction length, prediction length]
        inp_mask = generate_mask(PREDICT, LOOKBACK, DEVICE)
        # target mask: [prediction length, prediction length]
        target_mask = generate_mask(PREDICT, PREDICT, DEVICE)
        model_out, _ = model.forward(x, target, inp_mask, target_mask)
    else:
        raise Exception("Bad model selected!")
    return model_out


def train_epoch(dl, epoch):
    print_once = True
    model.train(True)

    epoch_train_loss = 0.
    # loop over training batches
    times_run = 0

    for i, (x, target, y) in enumerate(dl):
        optim.zero_grad()  # zero gradients
        x = x.to(DEVICE)
        target = target.to(DEVICE)
        y = y.to(DEVICE)

        model_out = get_model_pred(x, target, y)
        # squeeze the tensors to account for 1 dim sizes
        model_out = model_out.squeeze()
        y = y.squeeze()
        if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
            y = torch.t(y)
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
    for i, (x, target, y) in enumerate(dl):
        # model_out = get_model_pred(x, target, y)
        if ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
            model_out = time_predict(model, x)
        else:
            model_out = get_model_pred(x, target, y)

        model_out = model_out.squeeze()
        y = y.squeeze()
        if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
            y = torch.t(y)

        loss = loss_func(model_out, y)
        overall_acc, overall_bias, _ = calc_all_accuracy(prediction=model_out, actual=y)
        epoch_test_loss += loss.item() * x.size(0)
        times_run += x.size(0)

        if ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
            return epoch_test_loss / times_run, overall_acc, overall_bias

    return epoch_test_loss / times_run, format(overall_acc, '.2f'), format(overall_bias, '.2f')


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

    avg_valid_loss, valid_overall_acc, valid_overall_bias = test_epoch(valid_dl, e)

    _, train_overall_acc, train_overall_bias = test_epoch(train_dl, e)
    if run_ml_flow == RUN_TYPE.MLFLOW_RUN:
        mlflow.log_metric("validation overall bias", valid_overall_bias, step=e)
        mlflow.log_metric("validation overall accuracy", valid_overall_acc, step=e)

        mlflow.log_metric("train overall bias", train_overall_bias, step=e)
        mlflow.log_metric("train overall accuracy", train_overall_acc, step=e)

    num_epochs_run += 1
    train_loss.append(avg_train_loss)
    valid_loss.append(avg_valid_loss)
    scheduler.step()
    if run_ml_flow == RUN_TYPE.MLFLOW_RUN:
        mlflow.log_metric("avg train loss", avg_train_loss, step=e)
        mlflow.log_metric("avg validation loss", avg_valid_loss, step=e)
    print(f"epoch {e}: avg train loss: {avg_train_loss} avg val loss: {avg_valid_loss}")
    print(f"\ttrain accuracy: {train_overall_acc}\tbias: {train_overall_bias}")
    print(f"\tvalid accuracy: {valid_overall_acc}\tbias: {valid_overall_bias}")
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
    print("=====saving param to MLflow=====")
    mlflow.log_params(train_run_params)
    mlflow.log_params(DATA_PREP_DICT)
    mlflow.log_params(MODEL_PARAM_DICT)
    print("saving model to MLflow...")
    model_uri = mlflow.get_registry_uri()
    mlflow.pytorch.log_model(model, MODEL_SAVE_PATH)
    mlflow.end_run()
    # mlflow.pytorch.save_model(model, MODEL_SAVE_PATH)
    # mlflow.register_model(f'runs:/{run_id}/{MODEL_CHOICE}', model)

print("=====saving locally=====")
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
