from visualization import *
from saving_reading_data import *

# dict_train_data, dict_valid_data, dict_test_data = read_dicts_from_fp()
# input_transformations, output_transformations = read_transformations_from_fp()
# model = read_model_from_fp()
# reg_data, transformed_data = read_data_from_fp()
#
# for key, val in enumerate(dict_test_data):
#     print(val)
# print()
# for key, val in enumerate(dict_train_data):
#     print(val)
# print()
# for key, val in enumerate(dict_valid_data):
#     print(val)
#
# show_all_model_prediction(dict_test_data, transformed_data, output_transformations, model, PREDICT_DISPLAY_COUNT)
#
# plot_train_val_loss(train_loss, valid_loss)
# plt.show()

#
# def eval_test_data():
#     for key, val in enumerate(dict_test_data):
#         print(val)


def calc_accuracy(prediction, actual):
    # Abs.Error = absolute(actual – forecast)
    # Accuracy = 1 - (Abs.Error / Actual)
    # Bias = (Forecast - Actual) / Actual

    # will check only up to the max elements present
    max_index = max(len(prediction), len(actual))
    # get the absolution error for each element
    all_abs_errors = 0.
    all_accuracy = 0.
    all_bias = 0.
    print(prediction.shape)
    print(actual.shape)
    # am monkey bran - working with tensors, not actual matrices - need to finish up eval function
    print(len(prediction))
    print(len(actual))
    for i in range(max_index):
        current_abs_err = abs(actual[i] - prediction[i])
        all_abs_errors += current_abs_err
        all_accuracy += 1 - current_abs_err / actual[i]
        all_bias += (prediction[i] - actual[i]) / actual[i]
    return all_abs_errors, all_accuracy,all_bias

