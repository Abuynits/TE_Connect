import mlflow

from model_constants import *
from data_processing import *
from time_transformer import time_predict
from visualization import *
from eval import *

run_ml_flow = True if mlflow.active_run() is True else False


# TOOD: undo to the point where had the show_all model prediction working for seq2seq
#
def show_all_model_prediction(pred_dict, transformed_data, output_transformations, model, count=10):
    for key, val in enumerate(pred_dict):
        count -= 1
        if count == 0:
            return  # account if a person wants to break out of trianing and evaluating

        print(val)
        x, _, y = prep_data_for_transformer_model(transformed_data[val], LOOKBACK, PREDICT, INPUT_DATA_COLS,
                                                  OUTPUT_DATA_COLS)
        print(val)
        print(x.shape)
        # print(trg.shape) not use lol
        print(y.shape)
        index_graphing = 0

        all_pred_data = np.squeeze(
            output_transformations[val].inverse_transform(
                transformed_data[val][OUTPUT_DATA_COLS])[0:LOOKBACK]).tolist()
        all_actual_data = np.squeeze(
            output_transformations[val].inverse_transform(
                transformed_data[val][OUTPUT_DATA_COLS])[0:len(x)])
        all_pred_data = []
        run_once = False
        # x_axis = np.array(transformed_data[val]["year_week_ordered"][i+len(x[1]):i+len(x[1])+Data_Prep.predict])
        for i in range(len(x)):

            if PREDICT_RECURSIVELY:
                if not run_once:
                    print("run once")
                    model_inp = torch.from_numpy(x[i]).float().to(DEVICE)
                    run_once = True
                else:
                    print("caution might be broken lol")
                    print("concatenating")
                    model_inp = np.concatenate(model_inp.detach().numpy(), pred.detach().numpy())
                    print(model_inp)
                    model_inp = model_inp[PREDICT:]
                    print(model_inp)
            else:
                model_inp = torch.from_numpy(x[i]).float().to(DEVICE)

            actual_model_out = torch.from_numpy(y[i]).float().to(DEVICE)

            # print(model_inp.shape)
            if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:

                pred = model(model_inp[None, :])
                pred = torch.unsqueeze(pred, 1)
            elif ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
                # pred = predict_tensor_seq_to_seq(model, model_inp, Data_Prep.predict)
                pred = model.predict_seq(model_inp)
            elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
                pred = time_predict(model, model_inp)
                pred = torch.squeeze(pred)

            # print("pred shape:", pred.shape)
            # print("val:",val)
            # print(output_transformations[val])
            pred_inv_t = output_transformations[val].inverse_transform(pred.detach().cpu())
            actual_model_inv_t = output_transformations[val].inverse_transform(actual_model_out.detach().cpu())

            # actual_in_t = reg_data[val]["sales_amount"][i + len(x[1]):i+len(x[1])+Data_Prep.predict]
            if ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
                all_pred_data.append(np.squeeze(pred_inv_t, 1)[0])
            if ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
                squeezed_arr = np.squeeze(pred_inv_t)[0]
                print(i, squeezed_arr.shape)
                all_pred_data.append(squeezed_arr)
            # all_actual_data.append(actual_in_t)
            # print(actual_in_t.shape)
            if VISUALIZATION_VERBOSE:
                print("actual pred data:", pred)
                print("actual pred data:", pred_inv_t)
                print("actual data:", actual_model_inv_t)
                print("shape", y[i].shape)

            # plt.plot(x_axis, pred_inv_t.T[0], label="pred 0")
            # plt.plot(x_axis, input_transformations[val].inverse_transform(y[i]).T[0], label="act 0")
            # plt.plot(x_axis,pred_inv_t.T[1],label="pred 1")
            # plt.plot(x_axis,transformations[val].inverse_transform(y[i]).T[1],label = "act 1")
            # plt.plot(x_axis,pred_inv_t.T[2],label="pred 2")
            # plt.plot(x_axis,transformations[val].inverse_transform(y[i]).T[2],label = "act 2")

            if PREDICT_MODEL_FORCAST and random.random() > PERCENT_DISPLAY_MODEL_FORCAST:
                eval_plot_acc_pred_bias(
                    f'Individual acc/bias & prediction: {val}',
                    pred_inv_t,
                    actual_model_inv_t,
                    file_name=f"indiv_acc_bias{val}",
                    index_graphing=None)
        if PREDICT_ALL_FORCAST:
            overall_acc, overall_bias = eval_plot_acc_pred_bias(
                f'Total acc/bias & prediction: {val}',
                all_pred_data,
                all_actual_data,
                file_name=f"total_acc_bias{val}",
                index_graphing=None)
            if run_ml_flow:
                mlflow.log_metric("overall accuracy:", overall_acc)
                mlflow.log_metric("overall bias:", overall_bias)


def eval_plot_acc_pred_bias(fig_title, pred_data, actual_data, file_name=None, index_graphing=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 8)
    fig.suptitle(fig_title)

    overall_acc, overall_bias, individual_acc, individual_bias, individual_abs_err = eval_data_prediction(pred_data,
                                                                                                          actual_data)
    ax2.plot(individual_acc, label="accuracy")
    ax2.plot(individual_bias, label="bias")
    ax2.set_title(f"acc:{overall_acc:.4f}, bias:{overall_bias:.4f}")
    ax2.legend()
    ax2.set_ylabel("percentage")
    ax2.set_xlabel("time steps")

    x_axis = list(range(0, len(actual_data)))
    x_axis_offset = list(range(PREDICT, len(pred_data) + PREDICT))
    cut_axis_offset = list(range(PREDICT, len(pred_data)))
    ax1.plot(x_axis_offset, pred_data, label="pred")
    ax1.plot(x_axis, actual_data, label="actual")
    ax1.plot(cut_axis_offset, individual_abs_err, label="abs err")
    ax1.set_ylabel("price (in $)")
    ax1.set_xlabel("time steps")
    ax1.legend()

    if index_graphing is not None:
        ax1.set_title(f"{INPUT_DATA_COLS[index_graphing]} predicted vs actual")
        ax1.set_ylabel(INPUT_DATA_COLS[index_graphing])
    else:
        ax1.set_title(f"predicted vs actual")
    ax1.set_xlabel("time steps")
    if SAVE_EVAL_PLOTS and file_name is not None:
        full_fp = f'{EVAL_PLOTS_FILE_PATH}/{file_name}.png'
        if not os.path.exists(EVAL_PLOTS_FILE_PATH):
            os.mkdir(EVAL_PLOTS_FILE_PATH)

        plt.savefig(full_fp, dpi=EVAL_PLOTS_DPI, bbox_inches='tight')
        if run_ml_flow:
            full_fp = "data_pred.png"
            mlflow.log_figure(plt, full_fp)
    plt.show()
    return overall_acc, overall_bias


def plot_train_val_loss(train_loss, valid_loss):
    plt.plot(range(0, len(train_loss)), train_loss, label="train loss")
    plt.plot(range(0, len(train_loss)), valid_loss, label="validation loss")
    plt.legend()
    plt.title("Train and Valid Loss for epochs")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()


def display_train_test_valid_data(all_train_data, all_valid_data, all_test_data):
    print("all_train_data:", len(all_train_data))
    print("all_valid_data:", len(all_valid_data))
    print("all_test_data:", len(all_test_data))


def display_group_df(grouped_df, limit=5):
    count = 0
    for name, group in grouped_df:
        print(name)
        print(len(group))
        display(group.head(10))
        # display(group.head())
        count = count + 1
        if count == limit:
            break


def check_data_transformations(check_transforms_key, reg_data, transformed_data, output_transformations):
    plt.title("Data Input")
    plt.plot(reg_data[check_transforms_key]['sales_amount'])
    plt.show()
    plt.title("Normalized Data")
    plt.plot(transformed_data[check_transforms_key]['sales_amount'])
    plt.show()
    transform = output_transformations[check_transforms_key]
    plt.title("un-Normalized Transform Data")
    inv_t_data = transform.inverse_transform(transformed_data[check_transforms_key][OUTPUT_DATA_COLS])
    # display(inv_t_data)
    # print(np.asarray(inv_t_data).shape)
    plt.plot(inv_t_data)
    plt.show()


def show_model_inp_out_shapes(train_x, train_tg, train_y, valid_x, valid_tg, valid_y, test_x, test_tg, test_y):
    print()
    print("x train shape:", train_x.shape)
    print("target train shape:", train_tg.shape)
    print("y train shape:", train_y.shape)
    print()
    print("x test shape:", test_x.shape)
    print("target test shape:", test_tg.shape)
    print("y test shape:", test_y.shape)
    print()
    print("x valid shape:", valid_x.shape)
    print("target valid shape:", valid_tg.shape)
    print("y valid shape:", valid_y.shape)
