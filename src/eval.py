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
    # print(prediction.shape)
    # print(actual.shape)
    # am monkey bran - working with tensors, not actual matrices - need to finish up eval function
    # print(len(prediction))
    # print(len(actual))

    ones = torch.ones(prediction.shape[0], prediction.shape[1])
    if torch.cuda.is_available():
        ones = ones.cuda()
    # print(ones.shape)
    # compute absolute error for each component
    individual_abs_err = torch.abs(torch.sub(actual, prediction))
    # compute accuracy for each component
    individual_acc = torch.sub(ones, torch.div(individual_abs_err, actual))

    individual_bias = torch.div((torch.sub(prediction, actual)), actual)

    # compute all actual sales by taking the sum of a tensor
    all_abs_error = torch.sum(individual_abs_err)
    all_actual_sales = torch.sum(actual)
    all_forcasted_sales = torch.sum(prediction)

    overall_acc = 1 - all_abs_error / all_actual_sales
    overall_bias = (all_forcasted_sales - all_actual_sales) / all_actual_sales
    return overall_acc.item(), overall_bias.item(), (individual_acc, individual_bias, individual_abs_err)
