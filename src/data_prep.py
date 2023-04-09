from src.utils.saving_reading_data import *
from src.utils.data_processing import *

reg_data, transformed_data = read_data_from_fp()

# get dicts, and arrays for the split data
dict_train_data = {}
dict_valid_data = {}
dict_test_data = {}

all_valid_data = []
all_train_data = []
all_test_data = []

if SPLIT_TYPE == DATA_SPLIT.ON_PRODUCT_CODES:
    split_data(transformed_data,
               dict_train_data,
               dict_valid_data,
               dict_test_data,
               all_valid_data,
               all_train_data,
               all_test_data)
else:
    split_each_data_group(transformed_data,
                          dict_train_data,
                          dict_valid_data,
                          dict_test_data,
                          all_valid_data,
                          all_train_data,
                          all_test_data)

display_train_test_valid_data(all_train_data, all_valid_data, all_test_data)

valid_x, valid_tg, valid_y = get_all_data_arr(all_valid_data)
train_x, train_tg, train_y = get_all_data_arr(all_train_data)
test_x, test_tg, test_y = get_all_data_arr(all_test_data)

show_model_inp_out_shapes(train_x, train_tg, train_y,
                          valid_x, valid_tg, valid_y,
                          test_x, test_tg, test_y)

# Save the dictionaries and arrays
save_train_val_test_dicts(dict_train_data, dict_valid_data, dict_test_data)
save_train_val_test_arrs(train_x, train_tg, train_y,
                         valid_x, valid_tg, valid_y,
                         test_x, test_tg, test_y)
# all_train_data = np.array(all_train_data)
# all_test_data = np.array(all_test_data)
# all_valid_data = np.array(all_valid_data)

# save data:

print("Save data params....")
save_json(DATA_PREP_DICT, DATA_PARAM_FILE_PATH)
