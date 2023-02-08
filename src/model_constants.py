from data_constants import *

ML_FLOW_EXPERIMENT_NAME = "testing"
# MODEL INFO:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
LEARNING_RATE = 1e-2
EPOCHS = 50

TRAIN_PREDICTION = 'recursive'

GAMMA = 0.99


class MODEL_CHOICE(Enum):
    SEQ2SEQ = 0
    BASIC_LSTM = 1
    TIME_TRANSFORMER = 2


ARCH_CHOICE = MODEL_CHOICE.TIME_TRANSFORMER

LSTM_INP_SIZE = INPUT_DATA_FEATURES  # number of input features in lstm
LSTM_OUT_SIZE = OUTPUT_DATA_FEATURES  # number of output features in lstm
LSTM_HIDDEN_SIZE = 128  # number of lstm cells
LST_DROPOUT = 0.  # dropout
LSTM_LAYER_COUNT = 2  # number of layers in lstm cell

SEQ2SEQ_ENCODER_DROPOUT = 0.25
SEQ2SEQ_DECODER_DROPOUT = 0.25
SEQ2SEQ_HIDDEN_SIZE = 128  # number of lstm cells
SEQ2SEQ_DROPOUT = 0.  # dropout
SEQ2SEQ_LAYER_COUNT = 2  # number of layers in lstm cell
SEQ2SEQ_INPUT_SEQ_LENGTH = LOOKBACK  # length of input sequence to LSTM

TIME_MAX_SEQ_LEN = 5000  # hyper parameter for initialization of positional encoder
TIME_POS_ENC_DROP = 0.0
TIME_ENC_DROP = 0.0  # default 0.1
TIME_DEC_DROP = 0.0  # default 0.1
TIME_ENC_DIM_VAL = 512  # default: 512
TIME_ENC_HEAD_COUNT = 4  # default: 4
TIME_ENC_LAYER_COUNT = 4  # default: 4

TIME_DEC_DIM_VAL = 512  # default: 512
TIME_DEC_HEAD_COUNT = 4  # default: 4
TIME_DEC_LAYER_COUNT = 4  # default: 4
TIME_DEC_DIM_FEED_FORWARD = 2048  # default: = 2048
TIME_ENC_DIM_FEED_FORWARD = 2048  # default: = 2048

MODEL_CHOICE_NAME = "lstm" if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM else "seq2seq"

MODEL_PARAM_DICT = {
    "GAMMA": GAMMA,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "MODEL_CHOICE": MODEL_CHOICE_NAME,
    "LSTM_INP_SIZE": LSTM_INP_SIZE,
    "LSTM_OUT_SIZE": LSTM_OUT_SIZE,
    "LSTM_HIDDEN_SIZE": LSTM_HIDDEN_SIZE,
    "LSTM_DROPOUT": LST_DROPOUT,
    "LSTM_LAYER_COUNT": LSTM_LAYER_COUNT,
    "SEQ2SEQ_ENCODER_DROPOUT": SEQ2SEQ_ENCODER_DROPOUT,
    "SEQ2SEQ_DECODER_DROPOUT": SEQ2SEQ_DECODER_DROPOUT,
    "SEQ2SEQ_HIDDEN_SIZE": SEQ2SEQ_HIDDEN_SIZE,
    "SEQ2SEQ_DROPOUT": SEQ2SEQ_DROPOUT,
    "SEQ2SEQ_LAYER_COUNT": SEQ2SEQ_LAYER_COUNT,
    "SEQ2SEQ_INPUT_SEQ_LENGTH": SEQ2SEQ_INPUT_SEQ_LENGTH

}
