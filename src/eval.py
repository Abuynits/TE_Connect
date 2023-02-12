from visualization import *
from saving_reading_data import *


def calc_all_accuracy(prediction, actual):
    # Abs.Error = absolute(actual – forecast)
    # Accuracy = 1 - (Abs.Error / Actual)
    # Bias = (Forecast - Actual) / Actual

    # will check only up to the max elements present
    max_index = max(len(prediction), len(actual))
    # get the absolution error for each element
    all_abs_errors = 0.
    all_accuracy = 0.
    all_bias = 0.
    print(prediction.dim())
    if prediction.dim() == 1:
        prediction = prediction[None,:]
        actual = actual[None,:]
    print(prediction.shape)
    print(actual.shape)
    # am monkey bran - working with tensors, not actual matrices - need to finish up eval function
    print(prediction.dim())
    print(len(prediction))
    print(len(actual))
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
    return overall_acc.item(), overall_bias.item(), \
        (individual_acc.squeeze().numpy(), individual_bias.squeeze().numpy(), individual_abs_err.squeeze().numpy())


def show_all_eval_data(pred_inv_t, actual_model_inv_t):
    print("eval data len:",len(pred_inv_t))
    print("actual data len:", len(actual_model_inv_t))
    overall_acc, overall_bias, \
        (individual_acc, individual_bias, individual_abs_err) = calc_all_accuracy(
        torch.FloatTensor(pred_inv_t), torch.FloatTensor(actual_model_inv_t))
    print(f"Accuracy: {format(overall_acc, '.4f')}, Bias: {format(overall_bias, '.2f')}")


    plt.plot(individual_acc, label="accuracy")
    plt.plot(individual_bias, label="bias")
    plt.legend()
    plt.ylabel("percentage")
    plt.xlabel("time steps")
    plt.show()
    return individual_abs_err
