import pandas as pd
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
from pandas import DataFrame
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import datasets, transforms
import random as random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from torch.autograd import Variable
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from IPython.display import display
import mlflow
import dagshub
import json
import time
import os
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import random as andom
import math
from sklearn.compose import ColumnTransformer

# DATA INFO
# location of CSV file:
# TRACKED BY DVC!!!!
MODEL_SAVE_PATH = "data/model/model.pkl"
DATA_PARAM_FILE_PATH = "metrics/data_param.json"
MODEL_PARAM_FILE_PATH = "metrics/model_param.json"
MODEL_TRAIN_METRICS_FILE_PATH = "metrics/train_metric.json"

ML_RUNS_PATH = "mlruns"
# not tracked by dvc
READ_FROM_DRIVE = False
CSV_DRIVE_PATH = "content/gdrive/MyDrive/te_ai_cup_sales_forecasting_data.csv"
CSV_FILE_PATH = "data/te_ai_cup_sales_forecasting_data.csv"

TRAIN_DICT_FILE_PATH = "data/dicts/train_dict.pkl"
VAL_DICT_FILE_PATH = "data/dicts/validation_dict.pkl"
TEST_DICT_FILE_PATH = "data/dicts/test_dict.pkl"

TRAIN_X_FILE_PATH = "data/model_io/train_x.pkl"
VAL_X_FILE_PATH = "data/model_io/validation_x.pkl"
TEST_X_FILE_PATH = "data/model_io/test_x.pkl"
TRAIN_Y_FILE_PATH = "data/model_io/train_y.pkl"
VAL_Y_FILE_PATH = "data/model_io/validation_y.pkl"
TEST_Y_FILE_PATH = "data/model_io/test_y.pkl"

TRAIN_TARGET_FILE_PATH = "data/model_io/train_target.pkl";
VAL_TARGET_FILE_PATH = "data/model_io/validation_target.pkl";
TEST_TARGET_FILE_PATH = "data/model_io/test_target.pkl";

INPUT_TRANSFORMATIONS_FILE_PATH = "data/transformations/input_transformations.pkl"
OUTPUT_TRANSFORMATIONS_FILE_PATH = "data/transformations/output_transformations.pkl"
EVAL_PLOTS_FILE_PATH = "data/eval_plots"

FACTOR_TO_PRODUCT_DICT_FP = "data/factor_to_product.json"
PRODUCT_TO_FACTOR_DICT_FP = "data/product_to_factor.json"
MLFLOW_URL = "https://dagshub.com/Abuynits/TE_Connect.mlflow"

class DATA_SPLIT(Enum):
    ON_PRODUCT_CODES = 0  # split into test,validation, training for each product code
    ON_EACH_PRODUCT = 1  # get data from each product code


SPLIT_TYPE = DATA_SPLIT.ON_EACH_PRODUCT


class RUN_TYPE(Enum):
    MLFLOW_RUN = 0
    GIT_RUN = 1


EXPERIMENT_SOURCE = RUN_TYPE.MLFLOW_RUN

REGULAR_DATA_FILE_PATH = "data/regular_data.pkl"
TRANSFORMED_DATA_FILE_PATH = "data/transformed_data.pkl"
GROUP_BY_PRODUCT_LINE = True  # control whether you want to split the data on the product line groups
GROUP_BY_BUSINESS_UNIT = True  # control if you want to split the data on business units
GROUP_BY_COMPANY_REGION = True  # control if you want to split the data on regions
CHECK_DL = False  # debugging purposes for checking if dataloader works
CHECK_DATA_TRANSFORMS = False  # check if transforms for input and outputs of the model line up
SHOW_SAMPLE_TEST_TRAIN_VAL_SPLIT = True  # display a sample of the test-train-validation split
SAVE_MODEL = False  # control whether you want to save the model
DISPLAY_BATCH_TRAIN_DATA = False  # debug: check if you want to display the atches in training
LSTM_VERBOSE = False  # debug: check if you want to debug the LSTM model
SEQ2SEQ_VERBOSE = False  # debug: check if you want to debug the seq2seq model
TIME_PRED_VERBOSE = False
TIME_VERBOSE = False
EVAL_VERBOSE = False
EVAL_TRAIN_ACC = False
APPLY_LOG_TRANSFORM = False
VISUALIZATION_VERBOSE = False  # verbose mode for visualization
PREDICT_ALL_FORCAST = True  # prediction: check if you want to plot global predictions for all data points
PREDICT_MODEL_FORCAST = False  # prediction: check if you want to plot prediction for each prediction sement
PREDICT_RECURSIVELY = False  # control how you want to predict
PERCENT_DISPLAY_MODEL_FORCAST = 0.1  # display if greater than 0.9
PREDICT_DISPLAY_COUNT = 10  # number of times you will display the data
SAVE_EVAL_PLOTS = True
EVAL_PLOTS_DPI = 72