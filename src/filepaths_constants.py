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
from sklearn.preprocessing import MinMaxScaler
import random as andom

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
PREDICT_ALL_FORCAST = True  # prediction: check if you want to plot global predictions for all data points
PREDICT_MODEL_FORCAST = False  # prediction: check if you want to plot prediction for each prediction sement
PREDICT_RECURSIVELY = False  # control how you want to predict
PERCENT_DISPLAY_MODEL_FORCAST = 0.9  # display if greater than 0.9
PREDICT_DISPLAY_COUNT = 10  # number of times you will display the data

"""
TODO:
1. have a global config file (this one)
    will hold file paths
    debug control statements, and general booleans that control the run conditions
2. have a file for all data constants
    will store information regarding data collumn selection and so on
3. have a file for all the model constants
    will contain some identical inputs to the data constants and hold them the same for runs
4. have a constants file for evaluation part of the data
^^ done in order to prevent having to rerun everything after changing one constants file
===============
5. add eval.py and find a way of computing accuracy between graphs and predictions

6. add a way of saving plots in the library
7. add a boolean to control whether i use git for saving, mlflow, or both (dont know why)
- also need to add in mlflow integration and ease of use


8. need to update pipeline to save data processing parameters
- maybe add another step in pipeline - one to load data and have it ready, another to split it based on cols, features

9. need to add mlflow experiments to control the data available and the factors being studied

"""
