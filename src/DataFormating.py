from enum import Enum


class DataTypes(Enum):
    """Defines numerical types of each column."""
    QUANT = 0
    CAT = 1


class InputTypes(Enum):
    """Defines input types of each column."""
    TARGET = 0
    HIST_INPUT = 1
    FUTURE_HIST_INP = 2
    STATIC_INPUT = 3