from filepaths_constants import *
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