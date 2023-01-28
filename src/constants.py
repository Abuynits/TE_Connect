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
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from IPython.display import display
import mlflow
import dagshub
import json
import time
import os

# DATA INFO
# location of CSV file:
# TRACKED BY DVC!!!!
MODEL_SAVE_PATH = "data/model/model.pkl"
DATA_PARAM_FILE_PATH = "metrics/data_param.json"
MODEL_PARAM_FILE_PATH = "metrics/model_param.json"
MODEL_TRAIN_METRICS_FILE_PATH = "metrics/train_metric.json"
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

INPUT_TRANSFORMATIONS_FILE_PATH = "data/transformations/input_transformations.pkl"
OUTPUT_TRANSFORMATIONS_FILE_PATH = "data/transformations/output_transformations.pkl"

REGULAR_DATA_FILE_PATH = "data/regular_data.pkl"
TRANSFORMED_DATA_FILE_PATH = "data/transformed_data.pkl"

GROUP_BY_PRODUCT_LINE = True
GROUP_BY_BUSINESS_UNIT = True
GROUP_BY_COMPANY_REGION = True

# must contain each one of these labels
DATA_FILTER = ["Asia Pacific & ANZ", "Channel - Industrial"]
# data_filter = ["Asia Pacific & ANZ"]
# collumns of interest
INPUT_DATA_COLS = ["sales_amount", "sales_quantity",
                   "Price"]  # add features to end to make itself predict the output col - lstm not limit to 1 feature
# input_data_cols = ["sales_amount"]
OUTPUT_DATA_COLS = ["sales_amount"]
OUTPUT_DATA_FEATURES = len(OUTPUT_DATA_COLS)
INPUT_DATA_FEATURES = len(INPUT_DATA_COLS)
TEST_TRAIN_SPLIT = 0.8  # test-train split percentage
LOOKBACK = 10  # number of units used to make prediction
PREDICT = 5  # number of units that will be predicted
CHECK_DL = False  # debugging purposes for checking if dataloader works
CHECK_DATA_TRANSFORMS = False

SHOW_SAMPLE_TEST_TRAIN_VAL_SPLIT = True

# MODEL INFO:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
LEARNING_RATE = 5e-3
EPOCHS = 5

SAVE_MODEL = False
TRAIN_PREDICTION = 'recursive'
DISPLAY_BATCH_TRAIN_DATA = False
GAMMA = 0.9999


class MODEL_CHOICE(Enum):
    SEQ2SEQ = 0
    BASIC_LSTM = 1


ARCH_CHOICE = MODEL_CHOICE.SEQ2SEQ

LSTM_INP_SIZE = INPUT_DATA_FEATURES  # number of input features in lstm
LSTM_OUT_SIZE = OUTPUT_DATA_FEATURES  # number of output features in lstm
LSTM_HIDDEN_SIZE = 128  # number of lstm cells
LST_DROPOUT = 0.  # dropout
LSTM_LAYER_COUNT = 2  # number of layers in lstm cell
LSTM_VERBOSE = False

SEQ2SEQ_ENCODER_DROPOUT = 0.15
SEQ2SEQ_DECODER_DROPOUT = 0.15
SEQ2SEQ_HIDDEN_SIZE = 128  # number of lstm cells
SEQ2SEQ_DROPOUT = 0.  # dropout
SEQ2SEQ_LAYER_COUNT = 2  # number of layers in lstm cell
SEQ2SEQ_INPUT_SEQ_LENGTH = LOOKBACK  # length of input sequence to LSTM
SEQ2SEQ_VERBOSE = False

PREDICT_ALL_FORCAST = True
PREDICT_MODEL_FORCAST = False
PERCENT_DISPLAY_MODEL_FORCAST = 0.9  # display if greater than 0.9
PREDICT_RECURSIVELY = False
PREDICT_DISPLAY_COUNT = 10

MODEL_CHOICE_NAME = "lstm" if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM else "seq2seq"

DATA_PREP_DICT = {
    "GROUP_BY_PRODUCT_LINE": GROUP_BY_PRODUCT_LINE,
    "GROUP_BY_BUSINESS_UNIT": GROUP_BY_BUSINESS_UNIT,
    "GROUP_BY_COMPANY_REGION": GROUP_BY_COMPANY_REGION,
    "DATA_FILTER": DATA_FILTER,
    "INPUT_DATA_COLS": INPUT_DATA_COLS,
    "OUTPUT_DATA_COLS": OUTPUT_DATA_COLS,
    "OUTPUT_DATA_FEATURES": OUTPUT_DATA_FEATURES,
    "INPUT_DATA_FEATURES": INPUT_DATA_FEATURES,
    "TEST_TRAIN_SPLIT": TEST_TRAIN_SPLIT,
    "LOOKBACK": LOOKBACK,
    "PREDICT": PREDICT,
}

MODEL_PARAM_DICT = {
    "TRAINING_PARAM": {
        "GAMMA": GAMMA,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
    },
    "MODEL_PARAM": {
        "MODEL_CHOICE": MODEL_CHOICE_NAME,
        "LSTM_ARCH": {
            "LSTM_INP_SIZE": LSTM_INP_SIZE,
            "LSTM_OUT_SIZE": LSTM_OUT_SIZE,
            "LSTM_HIDDEN_SIZE": LSTM_HIDDEN_SIZE,
            "LST_DROPOUT": LST_DROPOUT,
            "LSTM_LAYER_COUNT": LSTM_LAYER_COUNT
        },
        "SEQ2SEQ_ARCH": {
            "SEQ2SEQ_ENCODER_DROPOUT": SEQ2SEQ_ENCODER_DROPOUT,
            "SEQ2SEQ_DECODER_DROPOUT": SEQ2SEQ_DECODER_DROPOUT,
            "SEQ2SEQ_HIDDEN_SIZE": SEQ2SEQ_HIDDEN_SIZE,
            "SEQ2SEQ_DROPOUT": SEQ2SEQ_DROPOUT,
            "SEQ2SEQ_LAYER_COUNT": SEQ2SEQ_LAYER_COUNT,
            "SEQ2SEQ_INPUT_SEQ_LENGTH": SEQ2SEQ_INPUT_SEQ_LENGTH
        }
    }
}
