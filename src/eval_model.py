from saving_reading_data import *
from data_processing import *
from eval import *

# check if mlflow exists and can be run
run_ml_flow = True if mlflow.active_run() is True else False
# create dataloader for test dataset
test_x, test_tg, test_y = read_test_data_from_fp()

test_ds = finance_data_set(test_x, test_tg, test_y)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"batches in test dl: {len(test_dl)}")
print(next(iter(test_dl))[0].shape)
print(next(iter(test_dl))[1].shape)
print(next(iter(test_dl))[2].shape)

dict_train_data, dict_valid_data, dict_test_data = read_dicts_from_fp()
input_transformations, output_transformations = read_transformations_from_fp()
model = read_model_from_fp()
reg_data, transformed_data = read_data_from_fp()

# display all data available
print("\n========dict_test_data========")
for key, val in enumerate(dict_test_data):
    print(val)

print("\n========dict_train_data========")
for key, val in enumerate(dict_train_data):
    print(val)

print("\n========dict_valid_data========")
for key, val in enumerate(dict_valid_data):
    print(val)

for key, val in enumerate(dict_test_data):
    x, tgt, y = prep_data_for_transformer_model(transformed_data[val], LOOKBACK, PREDICT, INPUT_DATA_COLS,
                                                OUTPUT_DATA_COLS)
    print()
    print(val)
    print(x.shape)
    # print(trg.shape) not use lol
    print(y.shape)
    # take inverse transformation
    all_pred_data = []
    print(transformed_data[val][OUTPUT_DATA_COLS].shape)
    all_actual_data = np.squeeze(
        output_transformations[val].inverse_transform(
            transformed_data[val][OUTPUT_DATA_COLS])[0:len(x)])
    print(np.shape(all_pred_data))
    print(np.shape(all_actual_data))
    for i in range(len(x)):
        # prepare data for input to model
        model_inp = torch.from_numpy(x[i]).float().to(DEVICE)
        # actual data output
        actual_model_out = torch.from_numpy(y[i]).float().to(DEVICE)
        # model prediction
        model_pred = get_model_prediction(model, model_inp)

        pred_inv_t = output_transformations[val].inverse_transform(model_pred.detach().cpu())
        actual_model_inv_t = output_transformations[val].inverse_transform(actual_model_out.detach().cpu())

        if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
            squeezed_arr = np.squeeze(pred_inv_t, 1)[0]
            all_pred_data.append(squeezed_arr)
        elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
            squeezed_arr = np.squeeze(pred_inv_t)[0]
            print(squeezed_arr)
            print(i)
            all_pred_data.append(squeezed_arr)
        elif ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:
            all_pred_data.append(pred_inv_t)

        if EVAL_VERBOSE:
            print("actual pred data:", model_pred)
            print("actual pred data:", pred_inv_t)
            print("actual data:", actual_model_inv_t)
            print("shape", y[i].shape)

        # display predictions for each product
        if PREDICT_MODEL_FORCAST and random.random() > PERCENT_DISPLAY_MODEL_FORCAST:
            if APPLY_LOG_TRANSFORM:
                pred_inv_t, _, _ = take_exponent(pred_inv_t)
                actual_model_inv_t, _, _ = take_exponent(actual_model_inv_t)
            eval_plot_acc_pred_bias(
                f'Individual acc/bias & prediction: {val}',
                pred_inv_t,
                actual_model_inv_t,
                file_name=f"indiv_acc_bias{val}",
                index_graphing=None)

    # display all forcast at the end
    if PREDICT_ALL_FORCAST:
        if APPLY_LOG_TRANSFORM:
            all_pred_data, _, _ = take_exponent(all_pred_data)
            all_actual_data, _, _ = take_exponent(all_actual_data)

        all_pred_data = torch.Tensor(all_pred_data)
        all_actual_data = torch.Tensor(all_actual_data)

        overall_acc, overall_bias = eval_plot_acc_pred_bias(
            f'Total acc/bias & prediction: {val}',
            all_pred_data,
            all_actual_data,
            file_name=f"total_acc_bias{val}",
            index_graphing=None)
        if run_ml_flow:
            mlflow.log_metric("overall accuracy:", overall_acc)
            mlflow.log_metric("overall bias:", overall_bias)

if run_ml_flow:
    mlflow.end_run()


def eval_test_data():
    for key, val in enumerate(dict_test_data):
        print(val)
