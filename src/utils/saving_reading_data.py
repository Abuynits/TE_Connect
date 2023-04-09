from ..config.data_constants import *
from ..config.filepaths_constants import *


def save_train_val_test_dicts(train_dict, val_dict, test_dict):
    with open(TRAIN_DICT_FILE_PATH, 'wb') as f:
        pickle.dump(train_dict, f)
    with open(VAL_DICT_FILE_PATH, 'wb') as f:
        pickle.dump(val_dict, f)
    with open(TEST_DICT_FILE_PATH, 'wb') as f:
        pickle.dump(test_dict, f)


def save_train_val_test_arrs(train_x, train_tg, train_y, valid_x, valid_tg, valid_y, test_x, test_tg, test_y):
    with open(TRAIN_X_FILE_PATH, 'wb') as f:
        pickle.dump(train_x, f)
    with open(VAL_X_FILE_PATH, 'wb') as f:
        pickle.dump(valid_x, f)
    with open(TEST_X_FILE_PATH, 'wb') as f:
        pickle.dump(test_x, f)

    with open(TRAIN_Y_FILE_PATH, 'wb') as f:
        pickle.dump(train_y, f)
    with open(VAL_Y_FILE_PATH, 'wb') as f:
        pickle.dump(valid_y, f)
    with open(TEST_Y_FILE_PATH, 'wb') as f:
        pickle.dump(test_y, f)

    with open(TRAIN_TARGET_FILE_PATH, 'wb') as f:
        pickle.dump(train_tg, f)
    with open(VAL_TARGET_FILE_PATH, 'wb') as f:
        pickle.dump(valid_tg, f)
    with open(TEST_TARGET_FILE_PATH, 'wb') as f:
        pickle.dump(test_tg, f)


def save_all_transformations(input_transformations, output_transformations):
    with open(INPUT_TRANSFORMATIONS_FILE_PATH, 'wb') as f:
        pickle.dump(input_transformations, f)
    with open(OUTPUT_TRANSFORMATIONS_FILE_PATH, 'wb') as f:
        pickle.dump(output_transformations, f)


def save_all_data(reg_data, transformed_data):
    with open(REGULAR_DATA_FILE_PATH, 'wb') as f:
        pickle.dump(reg_data, f)
    with open(TRANSFORMED_DATA_FILE_PATH, 'wb') as f:
        pickle.dump(transformed_data, f)


def read_transformations_from_fp():
    inp_transformation = {}
    out_transformation = {}
    with open(INPUT_TRANSFORMATIONS_FILE_PATH, 'rb') as f:
        inp_transformation = pickle.load(f)
    with open(OUTPUT_TRANSFORMATIONS_FILE_PATH, 'rb') as f:
        out_transformation = pickle.load(f)
    return inp_transformation, out_transformation


def read_data_from_fp():
    reg_data = {}
    transformed_data = {}
    with open(REGULAR_DATA_FILE_PATH, 'rb') as f:
        reg_data = pickle.load(f)
    with open(TRANSFORMED_DATA_FILE_PATH, 'rb') as f:
        transformed_data = pickle.load(f)
    return reg_data, transformed_data


def read_dicts_from_fp():
    with open(TRAIN_DICT_FILE_PATH, 'rb') as f:
        train_dict = pickle.load(f)
    with open(VAL_DICT_FILE_PATH, 'rb') as f:
        val_dict = pickle.load(f)
    with open(TEST_DICT_FILE_PATH, 'rb') as f:
        test_dict = pickle.load(f)
    return train_dict, val_dict, test_dict


def read_test_data_from_fp():
    with open(TEST_X_FILE_PATH, 'rb') as f:
        test_x = pickle.load(f)
    with open(TEST_TARGET_FILE_PATH, 'rb') as f:
        test_tg = pickle.load(f)
    with open(TEST_Y_FILE_PATH, 'rb') as f:
        test_y = pickle.load(f)
    return test_x, test_tg, test_y


def read_train_arrs_from_fp():
    with open(TRAIN_X_FILE_PATH, 'rb') as f:
        train_x = pickle.load(f)
    with open(VAL_X_FILE_PATH, 'rb') as f:
        valid_x = pickle.load(f)

    with open(TRAIN_Y_FILE_PATH, 'rb') as f:
        train_y = pickle.load(f)
    with open(VAL_Y_FILE_PATH, 'rb') as f:
        valid_y = pickle.load(f)

    with open(TRAIN_TARGET_FILE_PATH, 'rb') as f:
        train_tg = pickle.load(f)
    with open(VAL_TARGET_FILE_PATH, 'rb') as f:
        valid_tg = pickle.load(f)

    return train_x, train_y, train_tg, valid_x, valid_y, valid_tg


def save_model(model):
    check_if_file_exists(MODEL_SAVE_PATH)
    torch.save(model, MODEL_SAVE_PATH)


def save_json(dict, filepath):
    check_if_file_exists(filepath)
    with open(filepath, 'w') as fp:
        json.dump(dict, fp)


def read_model_from_fp():
    model = torch.load(MODEL_SAVE_PATH)
    return model


def check_if_file_exists(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def reset_output_file():
    with open(RESULTS_FILE_PATH, 'w') as f:
        writer = csv.writer(f)
        header_row = ["year", "month", "fiscal week", "id", "business group", "region", "pred", "actual", "bias",
                      "abs_err"]
        writer.writerow(header_row)


def write_results_to_file(key,
                          all_pred_acc,
                          all_pred_bias,
                          acc,
                          bias,
                          abs_err,
                          monthly_acc,
                          monthly_bias,
                          pred_data,
                          actual_data,
                          time,
                          months):

    product_id = key[0]
    business_group = key[2]
    region = key[1]
    if len(actual_data) == 0 or len(pred_data) == 0 or len(abs_err) == 0 or len(bias) == 0:
        print("returning bc bad data")
        return
    with open(RESULTS_FILE_PATH, 'a') as f:
        writer = csv.writer(f)


        if PREDICTION_TYPE == prediction_time.DAILY:
            for row in range(len(pred_data)):
                fiscal_week = time[row] % 100
                year = time[row] // 100
                month = months[row]

                data_row = [year, month, fiscal_week, product_id, business_group, region, pred_data[row],
                            actual_data[row], bias[row], abs_err[row]]
                writer.writerow(data_row)
        elif PREDICTION_TYPE == prediction_time.MONTHLY:
            past_month = months[0]
            net_pred = 0
            net_actual = 0
            net_bias = 0
            net_abs_err = 0
            count = 0
            for row in range(len(pred_data)):
                fiscal_week = time[row] % 100
                year = time[row] // 100
                month = months[row]

                if past_month == month:
                    count += 1
                    net_actual += actual_data[row].item()
                    net_bias += bias[row]
                    net_abs_err += abs_err[row]
                    net_pred += pred_data[row].item()
                else:
                    data_row = [year, month, "-", product_id, business_group, region, net_pred / count,
                                net_actual / count, net_bias / count, net_abs_err / count]
                    writer.writerow(data_row)

                    net_pred = 0
                    net_actual = 0
                    net_bias = 0
                    net_abs_err = 0
                    count = 0
                    past_month = month
                    row = row - 1
        print(monthly_acc, monthly_bias)
        metadata_header = ["key", "all_pred_acc", "all_pred_bias", "monthly_acc", "monthly_bias"]
        writer.writerow(metadata_header)
        metadata = [key, all_pred_acc, all_pred_bias, monthly_acc, monthly_bias]
        writer.writerow(metadata)



"""
Try for the entire dataset - might not get same accuracy for dataset
try to run through all dataset - not cherry pick at the beginning
the accuracy might not be the actual picture
- expect it to be less than 80% in most cases
- this should be easy to do
take some time - forcasting for seq2seq

also break into smaller datasets - what are the computation thresholds
can ask for engagement with TE - agreement with university
if cannot run for all dataset - break into different BU and regions
- can use instance for 3 hrs and 
- once generate the file that he asks - generate the file, share it,
- we can have a better undrestanding of what hte model actually does

next thing:
- everyone needs to work on external indicator - a month from now on
- have a month to come up with them
- project not evaluated just best on the model
- model will depend on external indicators
- if go back to project definition - we coming up with external indicators that capture trend
can we capture the downstring or upstring that will happen
- use those external indicators and get those trends captured
- upstrings and downstrings - will boost model performance as well and outlook when present

they will be more interested in the indicators than the model

current econ: a lot of factors that impact the sales - a lot of prediction issues experienced
expect ~80% accuracy for the data
- they will reduce inventory and reduce supply - these are things that will not be historical
    sales data - can only get from external data
- if not leverage external indicator - will not be able to leverage the sharp downturn

have inflection points in dataset where test it out - example: covid
- if use seq2seq without external indicators,  you can say that model not pick up downtrend during covid

model never experience it before in their data - these are factors get from external indicators
- are certain downturns that see in dataset
- if back test, see if model can capture it

if able to give at least some indicators for at least half BU: better job than just giving a best model
- without external indicators

- get dataset ready and do for all things
- computing accuracy at month level is better for me - remove uncertainty when do at week level
"""
