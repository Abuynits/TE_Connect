from visualization import *
from saving_reading_data import *
from data_constants import *


def calc_all_accuracy(prediction, actual):
    # print(prediction.shape)
    # print(actual.shape)
    prediction = prediction.squeeze()
    actual = actual.squeeze()
    prediction = prediction[:-PREDICT]
    actual = actual[PREDICT:]

    return _calc_tensor_acc(prediction, actual)


def calc_feature_similarity(prediction, actual):
    prediction = torch.tensor(prediction).squeeze()
    actual = torch.tensor(actual).squeeze()

    return _calc_tensor_acc(prediction, actual)


def _calc_tensor_acc(prediction, actual):
    assert len(prediction) == len(actual), "something went very wrong"

    prediction = prediction.squeeze()
    actual = actual.squeeze()

    if torch.cuda.is_available():
        actual = actual.cuda()
        prediction = prediction.cuda()

    # print(prediction.is_cuda)
    # print(actual.is_cuda)
    ones = torch.ones_like(prediction)

    if torch.cuda.is_available():
        ones = ones.cuda()
    if EVAL_VERBOSE:
        print("actual_shape:", actual.shape)
        print(actual)
        print("prediction_shape:", prediction.shape)
        print(prediction)
    # print(ones.shape)
    # compute absolute error for each component
    individual_abs_err = torch.abs(torch.sub(actual, prediction))
    if EVAL_VERBOSE:
        print("indiv abs err shape", individual_abs_err.shape)
        print(individual_abs_err)
    # compute accuracy for each component
    if torch.cuda.is_available():
        individual_acc = torch.sub(ones, torch.div(individual_abs_err, actual))
    else:
        individual_acc = torch.sub(ones, torch.div(individual_abs_err.detach().cpu(), actual.detach().cpu()))
    # WORKAROUND BC BAD ACCURACY
    torch.nn.functional.relu(individual_acc, inplace=True)
    if EVAL_VERBOSE:
        print("indiv acc shape", individual_acc.shape)
        print(individual_acc)
    individual_bias = torch.div((torch.sub(prediction, actual)), actual)
    if EVAL_VERBOSE:
        print("indiv bias shape", individual_bias.shape)
        print(individual_bias)
    # compute all actual sales by taking the sum of a tensor
    all_abs_error = torch.sum(individual_abs_err)
    if EVAL_VERBOSE:
        print("all abs err:", all_abs_error)
    torch.nn.functional.relu(actual, inplace=True)
    all_actual_sales = torch.sum(actual)
    if EVAL_VERBOSE:
        print("all actual sales:", all_actual_sales)
    all_forcasted_sales = torch.sum(prediction)
    if EVAL_VERBOSE:
        print("all forcasted sales:", all_forcasted_sales)
    overall_acc = 1 - all_abs_error / all_actual_sales
    overall_bias = (all_forcasted_sales - all_actual_sales) / all_actual_sales
    if EVAL_VERBOSE:
        print(overall_acc)
        print(overall_bias)
    return overall_acc.item(), overall_bias.item(), \
        (individual_acc.detach().squeeze().cpu().numpy(), \
         individual_bias.detach().squeeze().cpu().numpy(), \
         individual_abs_err.detach().squeeze().cpu().numpy())


def calc_train_accuracy(prediction, actual):
    if prediction.dim() == 1:
        prediction = prediction[None, :]
        actual = actual[None, :]
    return _calc_tensor_acc(prediction, actual)


def eval_data_prediction(pred_inv_t, actual_model_inv_t):
    # print("eval data len:", len(pred_inv_t))
    # print("actual data len:", len(actual_model_inv_t))
    overall_acc, overall_bias, \
        (individual_acc, individual_bias, individual_abs_err) = calc_all_accuracy(
        torch.FloatTensor(pred_inv_t), torch.FloatTensor(actual_model_inv_t))
    # print(f"Accuracy: {format(overall_acc, '.4f')}, Bias: {format(overall_bias, '.2f')}")

    return overall_acc, overall_bias, individual_acc, individual_bias, individual_abs_err
