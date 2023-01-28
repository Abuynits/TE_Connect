from data_constants import *
# MODEL INFO:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
LEARNING_RATE = 5e-3
EPOCHS = 5


TRAIN_PREDICTION = 'recursive'

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


SEQ2SEQ_ENCODER_DROPOUT = 0.15
SEQ2SEQ_DECODER_DROPOUT = 0.15
SEQ2SEQ_HIDDEN_SIZE = 128  # number of lstm cells
SEQ2SEQ_DROPOUT = 0.  # dropout
SEQ2SEQ_LAYER_COUNT = 2  # number of layers in lstm cell
SEQ2SEQ_INPUT_SEQ_LENGTH = LOOKBACK  # length of input sequence to LSTM




MODEL_CHOICE_NAME = "lstm" if ARCH_CHOICE == MODEL_CHOICE.BASIC_LSTM else "seq2seq"

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
