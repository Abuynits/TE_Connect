from eval import *
from collections import defaultdict
from filepaths_constants import *
run_ml_flow = True if mlflow.active_run() is True else False

def eval_plot_acc_pred_bias(fig_title, pred_data, actual_data, file_name=None, index_graphing=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 8)
    fig.suptitle(fig_title)
    # if ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
    #     pred_data = pred_data.squeeze() # TODO: need to fix visualization for transformer model
    print(pred_data.shape)
    print(actual_data.shape)
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


def display_factor_comparison(key, value, y1_var, y2_var):
    plt.title(key)
    plt.ylabel("standard difference")
    plt.xlabel("weekly timestep")
    plt.plot(value[y1_var], label=y1_var)
    plt.plot(value[y2_var], label=y2_var)
    plt.show()


def multi_dict(k, t):
    if k == 1:
        return defaultdict(t)
    else:
        return defaultdict(lambda: multi_dict(k - 1, t))
def display_multiple_factors_comparison(all_data,
                                        y1_var,
                                        y2_var,
                                        display_rows,
                                        display_cols,
                                        figure_width,
                                        figure_height,
                                        external_df=None):
    plt.rc('xtick', labelsize=6)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)
    fig, axs = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(figure_width, figure_height))

    # loop through tickers and axes
    x_loc, y_loc = 0, 0
    iteration_count = 0
    all_accuracy = []
    all_bias = []
    for key, val in all_data.items():
        if iteration_count != 0:
            if iteration_count == display_cols * (display_rows):
                y_loc = 0
                x_loc = 0
                iteration_count = 0
                plt.xticks(fontsize=6)
                avg_acc = sum(all_accuracy) / len(all_accuracy)
                avg_bias = sum(all_bias) / len(all_bias)
                sum_title_str = "{} vs {}\navg acc:{:.4f}    avg bias:{:.4f}".format(y1_var, y2_var,
                                                                                     avg_acc,
                                                                                     avg_bias)
                fig.suptitle(sum_title_str)
                fig.tight_layout(pad=2.0)
                leg = plt.legend(loc='upper right')
                plt.show()
                fig, axs = plt.subplots(nrows=display_rows, ncols=display_cols, figsize=(figure_width, figure_height))
            elif iteration_count % (display_rows) == 0:
                x_loc = 0
                y_loc += 1
        y1 = val[y1_var]
        # account for external factors
        if external_df is not None:
            y2 = external_df[y2_var]
        else:
            y2 = val[y2_var]
        actual_tensor = y1.values
        pred_tensor = y2.values
        overall_acc, overall_bias, (individual_acc, individual_bias, individual_abs_err) = calc_feature_similarity(
            pred_tensor,
            actual_tensor)
        all_accuracy.append(overall_acc)
        all_bias.append(overall_bias)
        axs[x_loc, y_loc].plot(y1, label="output")
        axs[x_loc, y_loc].plot(y2, label="external")
        axs[x_loc, y_loc].plot(individual_abs_err, label="abs err")
        title_str = "{}\nacc:{:.4f}    bias:{:.4f}".format(key, overall_acc, overall_bias)
        axs[x_loc, y_loc].set_title(title_str, fontsize=6)
        # logic for iteration

        iteration_count += 1
        x_loc += 1
    if iteration_count != 0:
        plt.xticks(fontsize=6)
        avg_acc = sum(all_accuracy) / len(all_accuracy)
        avg_bias = sum(all_bias) / len(all_bias)
        sum_title_str = "{} vs {}\navg acc:{:.4f}    avg bias:{:.4f}".format(y1_var, y2_var,
                                                                             avg_acc,
                                                                             avg_bias)
        fig.suptitle(sum_title_str)
        fig.tight_layout(pad=2.0)
        leg = plt.legend(loc='upper right')
        plt.show()


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
