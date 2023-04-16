from data_constants import *
ML_FLOW_EXPERIMENT_NAME = "testing"
# MODEL INFO:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EPOCHS = 5

GAMMA = 0.95


# control the type of model used for forcasting
class MODEL_CHOICE(Enum):
    SEQ2SEQ = 0
    BASIC_LSTM = 1
    TIME_TRANSFORMER = 2
    DEEP_ESN = 3
    TFT = 4


ARCH_CHOICE = MODEL_CHOICE.TFT

EARLY_STOP_MIN_EPOCH = 10
EARLY_STOP_DELTA = 0.05

LSTM_INP_SIZE = INPUT_DATA_FEATURES  # number of input features in lstm
LSTM_OUT_SIZE = OUTPUT_DATA_FEATURES  # number of output features in lstm
LSTM_HIDDEN_SIZE = 128  # number of lstm cells
LSTM_DROPOUT = 0.  # dropout
LSTM_LAYER_COUNT = 2  # number of layers in lstm cell

SEQ2SEQ_ENCODER_DROPOUT = 0.25
SEQ2SEQ_DECODER_DROPOUT = 0.25
SEQ2SEQ_HIDDEN_SIZE = 512  # 512 # number of lstm cells
SEQ2SEQ_LAYER_COUNT = 2  # 8 # number of layers in lstm cell
SEQ2SEQ_INPUT_SEQ_LENGTH = LOOKBACK  # length of input sequence to LSTM
SEQ2SEQ_MIXED_TEACHER_FORCING_RATIO = 0.5

ESN_HIDDEN_FEATURES = 5
ESN_NUM_LAYERS = 2
ESN_NON_LINEARITY = 'tanh'
ESN_BATCH_FIRST = False
ESN_LEAKING_RATE = 1
ESN_SPECTRAL_RADIUS = 0.9
ESN_W_IH_SCALE = 1
ESN_LAMBDA_REG = 0
ESN_DENSITY = 1
ESN_W_IO = False
ESN_READ_OUT_TRAINING = 'svd'
ESN_OUTPUT_STEPS = 'all'
ESN_WASHOUT_RATE = 0.0


class SEQ2SEQ_TRAIN_OPTIONS(Enum):
    GENERAL = 0
    TEACHER_FORCING = 1
    MIXED_TEACHER_FORCING = 2


SEQ2SEQ_TRAIN_TYPE = SEQ2SEQ_TRAIN_OPTIONS.TEACHER_FORCING

TIME_MAX_SEQ_LEN = 5000  # hyper parameter for initialization of positional encoder
TIME_POS_ENC_DROP = 0.0
TIME_BATCH_FIRST = True

TIME_ENC_DROP = 0.0  # default 0.1
TIME_ENC_DIM_VAL = 16  # default: 512
TIME_ENC_HEAD_COUNT = 1  # default: 4
TIME_ENC_LAYER_COUNT = 1  # default: 4
TIME_ENC_DIM_FEED_FORWARD = 16  # default: = 2048

TIME_DEC_DROP = 0.0  # default 0.1
TIME_DEC_DIM_VAL = 16  # default: 512
TIME_DEC_HEAD_COUNT = 1  # default: 4
TIME_DEC_LAYER_COUNT = 1  # default: 4
TIME_DEC_DIM_FEED_FORWARD = 16  # default: = 2048

TFT_TIME_STEPS = LOOKBACK
TFT_INPUT_SIZE = INPUT_DATA_FEATURES
TFT_OUTPUT_SIZE = OUTPUT_DATA_FEATURES
TFT_MULTIPROCESSING_WORKERS = 3  # TODO

if READ_FROM_DRIVE:
    from google.colab import drive

    drive.mount('/content/gdrive')
    df = pd.read_csv(CSV_DRIVE_PATH)
else:
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)

TFT_CATEGORY_COUNTS = [df["product_line_code"].unique(),
                       df["company_region_name_level_1"].unique(),
                       df["business_unit_group_name"].unique()]

TFT_INPUT_OBS_LOC = [i for i in INPUT_DATA_TYPE if i is InputTypes.FUTURE_HIST_INP]
TFT_STATIC_INPUT_LOC = [i for i in INPUT_DATA_TYPE if i is InputTypes.STATIC_INPUT]

TFT_REGULAR_INPUTS = [i for i in INPUT_DATA_TYPE if i is InputTypes.HIST_INPUT]
TFT_CATEGORICAL_INPUTS = [i for i in INPUT_DATA_FORMAT if i is DataTypes.CAT]

TFT_FUTURE_INPUTS = len([i for i in INPUT_DATA_TYPE if i is InputTypes.FUTURE_HIST_INP])
TFT_HIST_INPUTS = len([i for i in INPUT_DATA_TYPE if i is InputTypes.FUTURE_HIST_INP])

TFT_QUANTILES = [0.1, 0.5, 0.9]
TFT_HIDDEN_SIZE = 160
TFT_DROPOUT = 0.1
TFT_N_HEADS = 4
TFT_ENC_STEPS = PREDICT
TFT_STACKS = 1
TFT_N_CAT_VARS = len(TFT_CATEGORICAL_INPUTS)

# TFT_COL_DEF = [
#    ('id', DataTypes.REAL_VALUED, InputTypes.ID),
#    ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
#    ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
#    ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#    ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#    ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
#    ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
# ]

if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM:
    MODEL_CHOICE_NAME = "lstm"
elif ARCH_CHOICE == MODEL_CHOICE.TIME_TRANSFORMER:
    MODEL_CHOICE_NAME = "transformer"
elif ARCH_CHOICE == MODEL_CHOICE.TFT:
    MODEL_CHOICE_NAME = "tft"
else:
    MODEL_CHOICE_NAME = "seq2seq"

MODEL_PARAM_DICT = {
    "GAMMA": GAMMA,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "MODEL_CHOICE": MODEL_CHOICE_NAME,
    "LSTM_INP_SIZE": LSTM_INP_SIZE,
    "LSTM_OUT_SIZE": LSTM_OUT_SIZE,
    "LSTM_HIDDEN_SIZE": LSTM_HIDDEN_SIZE,
    "LSTM_DROPOUT": LSTM_DROPOUT,
    "LSTM_LAYER_COUNT": LSTM_LAYER_COUNT,
    "SEQ2SEQ_ENCODER_DROPOUT": SEQ2SEQ_ENCODER_DROPOUT,
    "SEQ2SEQ_DECODER_DROPOUT": SEQ2SEQ_DECODER_DROPOUT,
    "SEQ2SEQ_HIDDEN_SIZE": SEQ2SEQ_HIDDEN_SIZE,
    "SEQ2SEQ_DROPOUT": SEQ2SEQ_ENCODER_DROPOUT,
    "SEQ2SEQ_LAYER_COUNT": SEQ2SEQ_LAYER_COUNT,
    "SEQ2SEQ_INPUT_SEQ_LENGTH": SEQ2SEQ_INPUT_SEQ_LENGTH

}
