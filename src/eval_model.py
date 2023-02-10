from dl_ds import *
from filepaths_constants import *
from saving_reading_data import *
from model_constants import *
from visualization import *

# create dataloader for test dataset
test_x, test_tg, test_y = read_test_data_from_fp()

test_ds = finance_data_set(test_x, test_tg, test_y)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"batches in test dl: {len(test_dl)}")
print(next(iter(test_dl))[0].shape)
print(next(iter(test_dl))[1].shape)
print(next(iter(test_dl))[1].shape)

dict_train_data, dict_valid_data, dict_test_data = read_dicts_from_fp()
input_transformations, output_transformations = read_transformations_from_fp()
model = read_model_from_fp()
reg_data, transformed_data = read_data_from_fp()

for key, val in enumerate(dict_test_data):
    print(val)
print()
for key, val in enumerate(dict_train_data):
    print(val)
print()
for key, val in enumerate(dict_valid_data):
    print(val)

show_all_model_prediction(dict_test_data, transformed_data, output_transformations, model, PREDICT_DISPLAY_COUNT)

def eval_test_data():
    for key, val in enumerate(dict_test_data):
        print(val)
