from filepaths_constants import *

# must contain each one of these labels
DATA_FILTER = ["Channel - Industrial"]
# data_filter = ["Asia Pacific & ANZ"]
# collumns of interest
INPUT_DATA_COLS = ["sales_amount", "sales_quantity", "Price"]  # add features to end to make itself predict the output col - lstm not limit to 1 feature
# input_data_cols = ["sales_amount"]
OUTPUT_DATA_COLS = ["sales_amount"]
OUTPUT_DATA_FEATURES = len(OUTPUT_DATA_COLS)
INPUT_DATA_FEATURES = len(INPUT_DATA_COLS)

SPLIT_BY_PERCENT = False # True: use percentage below. False: remove last 76 data points for eval, use percent for val and train

TEST_TRAIN_SPLIT = 0.8  # test-train split percentage
PERCENT_TRAIN_DATA = 0.7
PERCENT_TEST_DATA = 0.1
PERCENT_VALID_DATA = 1 - PERCENT_TRAIN_DATA - PERCENT_TEST_DATA
LOOKBACK = 100  # number of units used to make prediction
PREDICT = 76  # number of units that will be predicted

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
