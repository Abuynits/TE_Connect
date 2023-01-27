from constants import *
from data_processing import prep_data_for_model

def show_all_model_prediction(pred_dict,transformed_data,output_transformations,model,count = 10):

    for key, val in enumerate(pred_dict):
        count -= 1
        if count == 0:
            return # account if a person wants tobreak out of trianign and evalutating

        print(val)
        x, y = prep_data_for_model(transformed_data[val], LOOKBACK, PREDICT, INPUT_DATA_COLS,
                         OUTPUT_DATA_COLS)
        print(val)
        print(x.shape)
        print(y.shape)
        index_graphing = 0

        all_pred_data = np.squeeze(
            output_transformations[val].inverse_transform(transformed_data[val][OUTPUT_DATA_COLS])[
            0:LOOKBACK]).tolist()
        all_actual_data = []
        verbose = False
        all_actual_data = np.squeeze(
            output_transformations[val].inverse_transform(transformed_data[val][OUTPUT_DATA_COLS])[0:len(x)])

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
            else:
                # pred = predict_tensor_seq_to_seq(model, model_inp, Data_Prep.predict)
                pred = model.predict_seq(model_inp)
            # print(pred)
            # print("pred shape:", pred.shape)
            # print(output_transformations[val])
            pred_inv_t = output_transformations[val].inverse_transform(pred.detach().cpu())
            actual_model_int_t = output_transformations[val].inverse_transform(actual_model_out.detach().cpu())
            # print("pred inv shape:", pred_inv_t.shape)

            # actual_in_t = reg_data[val]["sales_amount"][i + len(x[1]):i+len(x[1])+Data_Prep.predict]

            all_pred_data.append(np.squeeze(pred_inv_t, 1)[0])
            # all_actual_data.append(actual_in_t)
            # print(actual_in_t.shape)
            if verbose:
                print("actual pred data:", pred)
                print("actual pred data:", pred_inv_t)
                print("actual data:", actual_model_int_t)
                print("shape", y[i].shape)

            # plt.plot(x_axis, pred_inv_t.T[0], label="pred 0")
            # plt.plot(x_axis, input_transformations[val].inverse_transform(y[i]).T[0], label="act 0")
            # plt.plot(x_axis,pred_inv_t.T[1],label="pred 1")
            # plt.plot(x_axis,transformations[val].inverse_transform(y[i]).T[1],label = "act 1")
            # plt.plot(x_axis,pred_inv_t.T[2],label="pred 2")
            # plt.plot(x_axis,transformations[val].inverse_transform(y[i]).T[2],label = "act 2")
            if PREDICT_MODEL_FORCAST and random.random() > PERCENT_DISPLAY_MODEL_FORCAST:
                print("actual pred data:", pred)
                print("actual pred data:", pred_inv_t)
                print("shape", y[i].shape)
                plt.plot(pred_inv_t, label="pred")
                plt.plot(actual_model_int_t, label="actual")
                plt.legend()
                plt.title(f"{INPUT_DATA_COLS[index_graphing]} predicted vs actual")
                plt.ylabel(INPUT_DATA_COLS[index_graphing])
                plt.xlabel("time steps")
                plt.show()
        if PREDICT_ALL_FORCAST:
            plt.plot(all_pred_data, label="pred")
            plt.plot(all_actual_data, label="actual")
            plt.legend()
            plt.title(f"{INPUT_DATA_COLS[index_graphing]} predicted vs actual")
            plt.ylabel(INPUT_DATA_COLS[index_graphing])
            plt.xlabel("time steps")
            plt.show()

        # print(x.shape)
        # print(y.shape)
        # print(transformations[val])
def plot_train_val_loss(train_loss,valid_loss):
    plt.plot(range(0, len(train_loss)), train_loss, label="train loss")
    plt.plot(range(0, len(train_loss)), valid_loss, label="validation loss")
    plt.legend()
    plt.title("Train and Valid Loss for epochs")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()

def display_train_test_valid_data(all_train_data, all_valid_data, all_test_data):
    print("all_valid_data:", len(all_valid_data))
    print("all_train_data:", len(all_train_data))
    print("all_test_data:", len(all_test_data))

    print("all_valid_data shape:", all_valid_data.shape)
    print()
    print("all_test_data shape:", all_test_data.shape)
    print()
    print("all_train_data shape:", all_train_data.shape)
    print()

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

def show_model_inp_out_shapes(train_x,train_y,valid_x,valid_y,test_x,test_y):
    print()
    print("x train shape:", train_x.shape)
    print("y train shape:", train_y.shape)
    print()
    print("x test shape:", test_x.shape)
    print("y test shape:", test_y.shape)
    print()
    print("x valid shape:", valid_x.shape)
    print("y valid shape:", valid_y.shape)