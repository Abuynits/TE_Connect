import torch

from visualization import *
from model_constants import *
from visualization import *
from collections import defaultdict
from time_transformer import *


def calc_all_accuracy(prediction, actual):
    prediction = prediction.squeeze()
    actual = actual.squeeze()
    prediction = prediction[:-LOOKBACK]  # go from 100 -> end -> Lookback
    actual = actual[LOOKBACK:]  # go from 0 -> end
    print(prediction.shape)
    print(actual.shape)

    return _calc_tensor_acc(prediction, actual)


def calc_feature_similarity(prediction, actual):
    prediction = torch.FloatTensor(prediction).squeeze()
    actual = torch.FloatTensor(actual).squeeze()

    return _calc_tensor_acc(prediction, actual)


def get_model_prediction(model, model_inp):
    if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:

        pred = model(model_inp[None, :])
        pred = torch.unsqueeze(pred, 1)
    elif ARCH_CHOICE == MODEL_CHOICE.SEQ2SEQ:
        # pred = predict_tensor_seq_to_seq(model, model_inp, Data_Prep.predict)
        pred = model.predict_seq(model_inp)
    elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
        model_inp = model_inp.unsqueeze(0)
        # print("old imp shape:", model_inp.shape)
        # model_inp = torch.swapaxes(model_inp, 1, 2)
        # print("new imp shape:", model_inp.shape)
        pred = time_predict(model, model_inp)
        pred = torch.squeeze(pred)
        pred = pred.reshape(-1, 1)
    elif ARCH_CHOICE == MODEL_CHOICE.DEEP_ESN:
        model_inp = model_inp.unsqueeze(1)
        washout_list = [int(ESN_WASHOUT_RATE * model_inp.size(0))] * model_inp.size(1)
        pred, _ = model(model_inp, washout_list)
        pred = pred.view(pred.size(1), -1)
    else:
        raise Exception("error: invalid model selected")
    return pred


def multi_dict(k, t):
    if k == 1:
        return defaultdict(t)
    else:
        return defaultdict(lambda: multi_dict(k - 1, t))


def get_all_factor_comparison(all_data,
                              output_var,
                              external_data):
    product_to_indicator = multi_dict(3, str)
    indicator_to_product = multi_dict(3, str)

    for indicator in external_data:
        pred_tensor = external_data[indicator].values
        for key, val in all_data.items():
            actual_tensor = val[output_var].values
            if len(actual_tensor) < len(pred_tensor):
                pred_tensor = pred_tensor[0: len(actual_tensor)]
            elif len(actual_tensor) > len(pred_tensor):
                actual_tensor = actual_tensor[0: len(pred_tensor)]
            key = ','.join(key)
            actual_tensor = actual_tensor.squeeze()
            pred_tensor = pred_tensor.squeeze()

            overall_acc, overall_bias, (_, _, _) = calc_feature_similarity(pred_tensor, actual_tensor)
            indicator_to_product[indicator][key]['overall_acc'] = overall_acc
            indicator_to_product[indicator][key]['overall_bias'] = overall_bias

            product_to_indicator[key][indicator]['overall_acc'] = overall_acc
            product_to_indicator[key][indicator]['overall_bias'] = overall_bias

    return product_to_indicator, indicator_to_product


def _calc_tensor_acc(prediction, actual):
    if len(prediction) != len(actual):
        print("different lengths, skip calculation:", len(prediction), len(actual))
        # error return code
        return (-6), (-6), \
            (np.empty(shape=[len(prediction)]),
             np.empty(shape=[len(prediction)]),
             np.empty(shape=[len(prediction)]))

    prediction = prediction.squeeze()
    actual = actual.squeeze()
    actual = torch.nan_to_num(actual, nan=0, posinf=0, neginf=0)
    prediction = torch.nan_to_num(prediction, nan=0, posinf=0, neginf=0)
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
        individual_acc = torch.sub(ones, torch.div(individual_abs_err, torch.abs(actual)))
        individual_acc = torch.max(individual_acc, torch.tensor([0.]).cuda())
    else:
        individual_acc = torch.sub(ones, torch.div(individual_abs_err.detach().cpu(), torch.abs(actual).detach().cpu()))
        individual_acc = torch.max(individual_acc, torch.tensor([0.]))

    # clean up accuracy calculation
    individual_acc = torch.nan_to_num(individual_acc, nan=0, posinf=0, neginf=0)
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
    all_actual_sales = torch.sum(actual)
    if EVAL_VERBOSE:
        print("all actual sales:", all_actual_sales)
    all_forcasted_sales = torch.sum(prediction)
    if EVAL_VERBOSE:
        print("all forcasted sales:", all_forcasted_sales)
    overall_acc = 1 - abs(all_abs_error / all_actual_sales)
    overall_bias = (all_forcasted_sales - all_actual_sales) / all_actual_sales
    if EVAL_VERBOSE:
        print(overall_acc)
        print(overall_bias)
    return max(overall_acc.item(), 0.), overall_bias.item(), \
        (individual_acc.detach().squeeze().cpu().numpy(),
         individual_bias.detach().squeeze().cpu().numpy(),
         individual_abs_err.detach().squeeze().cpu().numpy())


def calc_train_accuracy(prediction, actual):
    if prediction.dim() == 1:
        prediction = prediction[None, :]
        actual = actual[None, :]
    return _calc_tensor_acc(prediction, actual)


def eval_data_monthly_pred(pred_inv_t, actual_inv_t, month_data):
    month_pred = []
    month_actual = []

    pred = pred_inv_t[len(actual_inv_t) - LOOKBACK - PREDICT:len(actual_inv_t) - LOOKBACK]
    actual = actual_inv_t[-PREDICT:]

    if len(actual) != len(pred):
        return -1, -1

    curr_month = month_data[0]
    prev_month = month_data[0]
    sum_pred = 0
    sum_actual = 0
    month_count = 0
    for loc in range(len(month_data)):
        if curr_month != prev_month:
            if month_count != 0:
                sum_pred /= month_count
                sum_actual /= month_count

            month_pred.append(sum_pred)
            month_actual.append(sum_actual)

            sum_actual = 0
            sum_pred = 0
            month_count = 0

        month_count += 1
        sum_pred += pred[loc]
        sum_actual += actual[loc]

        prev_month = curr_month
        curr_month = month_data[loc]

    month_overall_acc, month_overall_bias, (_, _, _) = calc_feature_similarity(
        month_pred,
        month_actual)
    return month_overall_acc, month_overall_bias


def eval_data_prediction(pred_inv_t, actual_inv_t):
    overall_acc, overall_bias, \
        (individual_acc, individual_bias, individual_abs_err) = calc_all_accuracy(
        torch.FloatTensor(pred_inv_t), torch.FloatTensor(actual_inv_t))

    pred_overall_acc, pred_overall_bias, (
        pred_individual_acc, pred_individual_bias, pred_individual_abs_err) = calc_feature_similarity(
        pred_inv_t[len(actual_inv_t) - LOOKBACK - PREDICT:len(actual_inv_t) - LOOKBACK],
        actual_inv_t[-PREDICT:]
    )
    # print(f"Accuracy: {format(overall_acc, '.4f')}, Bias: {format(overall_bias, '.2f')}")

    return (overall_acc, overall_bias), (pred_overall_acc, pred_overall_bias), \
        (individual_acc, individual_bias, individual_abs_err), \
        (pred_individual_acc, pred_individual_bias, pred_individual_abs_err)


def test_eval(pred, actual):
    return calc_feature_similarity(pred, actual)


pred = [-1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485, -1.51302485,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227, -1.06632227,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,  -0.6196197,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
        -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713, -0.17291713,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,  0.27378545,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,  0.72048802,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,   1.1671906,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,
         1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,  1.61389317,]


actual = [-1.25255961, -0.57168407, -0.57168407, -0.34472604, -0.59032927, -0.12741088,
          -0.43602314, -0.43602314, -0.59032927, -1.53212049, -1.45251934, -1.29331775,
          -1.85052437, -1.8787821 ,  0.44931328,  0.72320673,  0.44931328, -0.40439586,
          -0.40439586, -0.55285979, -0.84978838, -0.70132372, -1.0678423 , -1.0678423 ,
          -0.57698177,  1.01831406, -1.0331596 , -0.37816376,  0.27683281,  0.14583335,
           0.20840765,  0.20840765,  0.20840765,  0.39493467,  0.20840765,  0.38146657,
           0.18628603,  1.16218873,  3.89471556, -0.41254026, -0.27734006, -0.00693965,
           0.12826055, -0.43595237, -0.09304232, -0.26449734,  0.0784127 ,  0.24986773,
          -0.06402706, -0.06402706,  1.02800964,  3.21208376, -0.59383747, -0.03606468,
           0.10337815,  0.38226454, -0.14417415,  0.02472446, -0.14417415, -0.14417415,
          -0.31307277, -1.75048782, -1.75048782, -1.21128873, -1.75048782, -2.53392671,
          -0.5574931 ,  0.10131858, -0.1182858 , -0.12653781, -0.29631816, -0.63587816,
          -0.12653781,  0.38280327, -0.73103009, -0.59147387, -0.1728074 ,  1.50186138,
          -1.19652365,  0.03468652,  0.30828894,  0.44509015, -0.66432437, -0.81473594,
          -0.5139128 , -0.66432437, -0.06267808, -1.21334108, -0.97030944, -0.48424615,
           0.73091279, -0.77980771, -0.51863277,  0.13430422,  0.13430422, -0.88465667,
          -0.59160242, -0.73812954, -0.59160242, -0.00549463, -0.21776341,  0.17098728,
          -0.02338806,  0.55973796,  1.33723934, -0.87909539, -0.50151709, -0.12393879,
           0.12778032, -0.27084577, -0.0997251 , -0.0997251 , -0.44196643,  0.58475902,
          -1.3701639 , -1.19083239, -1.28049815, -1.3701639 , -0.77939681,  0.37544384,
           0.23108894,  0.51979874,  0.28912958,  0.71260412,  0.71260412,  0.50086685,
           1.13607867, -0.57766925, -0.57766925, -0.17615023,  0.62688853, -0.03510856,
          -0.45355261, -0.17458967,  0.38333549, -0.45121068,  0.2713618 ,  0.09071886,
           0.09071886,  0.45200474,  0.08484778,  0.24167349,  0.3984992 ,  0.86897632,
          -0.6194153 , -0.24079855, -0.36700365, -0.24079855, -0.28273597, -0.11224068,
          -0.45323125, -0.11224068,  0.56973974,  0.3993275 ,  0.93407309,  0.57757579,
           2.00356571, -0.46970494, -0.05346345,  0.08528396,  0.22403064, -1.05565457,
          -0.28487714, -0.59318826, -0.59318826,  0.1775899 , -1.10247084, -0.91634245,
          -1.19553503, -1.56779253, -0.14082988,  1.04263029,  0.87356414,  1.04263029,
           0.38696718,  0.57311579,  0.01467141, -0.35762509,  0.38696718, -0.07135255,
          -0.2093843 , -0.34741605,  0.34274272, -0.06706081,  0.71818562,  0.40408719,
           0.71818562, -0.63651293,  0.04245902,  0.04245902, -0.12728451,  0.04245902,
           0.0489952 ,  0.5359684 ,  0.21131936,  1.34759063, -0.36152542,  0.06946815,
           0.06946815,  0.35679648, -0.90638905, -0.76106936, -0.76106936, -0.47042998,
          -0.17979059, -0.60259065, -0.16465722,  0.12729793,  1.14914164, -0.77784563,
          -0.25512239, -0.1244414 ,  0.13692057, -0.05437697, -0.05437697,  0.31065109,
          -0.23689099,  0.49316512, -1.64192773, -0.92565529, -1.01518962, -1.4628598 ,
          -1.24627979,  0.74510088,  0.31837665,  0.31837665, -0.26101803, -0.42407444,
          -0.42407444, -0.42407444,  0.06509409, -1.06157043, -1.06157043, -0.83788157,
           0.16871684, -0.25384419, -0.12311193, -0.64604025, -0.12311193, -0.45703915,
           0.26416196,  0.44446188,  0.08386132,  0.26416196, -0.04878391, -0.1997992 ,
          -0.04878391,  1.31035658, -0.83809894, -0.60470438, -0.60470438, -0.48800639,
          -1.11597865, -0.98933809, -1.11597865, -1.11597865, -0.73605769, -1.04238073,
          -0.81694788, -0.92966395, -0.02793183, -0.80866482, -0.43864454, -0.31530493,
          -0.31530493, -0.54911544, -0.40046375, -0.40046375, -0.54911544, -0.8464181 ,
          -1.05330975, -0.8704332 , -0.8704332 , -2.05913113, -1.85223226,  0.05623764,
           0.1755166 , -0.54015934, -0.39284146, -0.04514139, -0.04514139, -0.21899178,
           0.30255868,  0.17927756,  0.02505159,  0.17927756,  0.95040668,  0.13145029,
          -0.54514145, -0.2745049 ,  0.13145029, -0.91363944, -0.76019773, -0.29987331,
          -0.45331502,  0.16045039, -1.06204921, -1.16900974, -0.74116763, -0.42028532,
          -1.64576885, -1.30461001, -1.30461001, -1.38990026, -0.85966883, -0.85966883,
          -0.99979916, -0.57940819, -0.43927787, -0.50572073, -0.8676912 , -0.74703437,
           0.21822095, -1.76582629, -1.51492393, -1.26402157, -1.26402157, -1.2036325 ,
          -0.83755227, -0.95957901, -1.2036325 , -1.2036325 , -1.61401447, -1.5433427 ,
          -1.75535728, -2.10871467, -2.35096783, -1.17979144, -1.17979144, -1.01248083,
          -0.28110391, -0.44315581, -0.76725963, -0.44315581,  0.36710445, -0.09229417,
          -0.37811537, -0.09229417,  0.76516798, -0.23051008, -0.08088998,  0.51758823,
           0.66720833,  0.52050644,  0.31837376,  0.31837376,  0.92477108,  0.7226384 ,
          -0.11880791,  0.17713422, -0.11880791,  0.76901848,  0.54779505,  0.38499716,
           0.71059221,  0.54779505,  0.68314546,  0.68314546, -0.15791234,  0.89340955,
           1.31393844,  1.10422102,  1.30536724,  1.50651347,  1.90880663,  0.86908753,
           0.51778899,  0.86908753,  1.0447368 ,  2.20692243,  2.20692243,  1.92046863,]

print()
print()
test_eval(pred, actual)