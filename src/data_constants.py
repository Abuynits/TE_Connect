from filepaths_constants import *
from DataFormating import DataTypes
from DataFormating import InputTypes

# must contain each one of these labels
DATA_FILTER = []
# data_filter = ["Asia Pacific & ANZ"]
# collumns of interest

# add features to end to make itself predict the output col - lstm not limit to 1 feature
INPUT_DATA_COLS = ["sales_amount",
                   "sales_quantity",
                   "Price",
                   "extr_1",
                   "extr_2",
                   "extr_3"]

# INPUT_DATA_COLS = ["sales_amount",
#                   "sales_quantity",
#                   "Price",
#                   "fiscal_week_historical",
#                   "business_unit_group_name",
#                   "company_region_name_level_1"]

# INPUT_NORM_COLS = ["sales_amount",
#                    "sales_quantity",
#                    "Price",
#                    "fiscal_week_historical",
#                    "fiscal_month_historical",
#                    "fiscal_quarter_historical",
#                    "fiscal_year_historical"]

INPUT_NORM_COLS = ["sales_amount",
                   "sales_quantity",
                   "Price",
                   "extr_1",
                   "extr_2",
                   "extr_3"]

# processed in this order
INPUT_DATA_FORMAT = [DataTypes.QUANT,
                     DataTypes.QUANT,
                     DataTypes.QUANT,
                     DataTypes.QUANT,
                     DataTypes.CAT,
                     DataTypes.CAT]

INPUT_DATA_TYPE = [InputTypes.HIST_INPUT,
                   InputTypes.HIST_INPUT,
                   InputTypes.HIST_INPUT,
                   InputTypes.FUTURE_HIST_INP,
                   InputTypes.STATIC_INPUT,
                   InputTypes.STATIC_INPUT]

OUTPUT_DATA_COLS = ["sales_amount"]

OUTPUT_NORM_COLS = ["sales_amount"]

OUTPUT_COL_FORMAT = [DataTypes.QUANT]
OUTPUT_COL_VAL = [InputTypes.TARGET]

TEST_TRAIN_SPLIT = 0.8  # test-train split percentage
PERCENT_TRAIN_DATA = 0.7
PERCENT_TEST_DATA = 0.1
PERCENT_VALID_DATA = 1 - PERCENT_TRAIN_DATA - PERCENT_TEST_DATA
LOOKBACK = 10  # number of units used to make prediction
PREDICT = 10  # number of units that will be predicted


class prediction_time(Enum):
    MONTHLY = 1
    DAILY = 2


PREDICTION_TYPE = prediction_time.DAILY
# input_data_cols = ["sales_amount"]
"""
fiscal_year_historical
fiscal_quarter_historical
fiscal_month_historical
fiscal_week_historical
business_unit_group_name
company_region_name_level_1
product_line_code
product_line_name
sales_quantity
sales_amount
year_week_ordered
Price
"""
# col_def = [
#     ('fiscal_year_historical', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('fiscal_quarter_historical', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('fiscal_month_historical', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('fiscal_week_historical', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('business_unit_group_name', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
#     ('company_region_name_level_1', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
#     ('product_line_code', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
#     ('product_line_name', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
#     ('sales_quantity', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('sales_amount', DataTypes.REAL_VALUED, InputTypes.TARGET),
#     ('year_week_ordered', DataTypes.DATE, InputTypes.TIME),
#     ('Price', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#     ('Id', DataTypes.CATEGORICAL, InputTypes.ID),
# ]

OUTPUT_DATA_FEATURES = len(OUTPUT_DATA_COLS)
INPUT_DATA_FEATURES = len(INPUT_DATA_COLS)

SPLIT_BY_PERCENT = False  # True: use percentage below. False: remove last 76 data points for eval, use percent for val and train

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
