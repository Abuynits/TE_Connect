from filepaths_constants import *


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
    torch.save(model,MODEL_SAVE_PATH)


def save_json(dict, filepath):
    check_if_file_exists(filepath)
    with open(filepath, 'w') as fp:
        json.dump(dict, fp)


def read_model_from_fp():
    model = torch.load(MODEL_SAVE_PATH)
    return model


def check_if_file_exists(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
