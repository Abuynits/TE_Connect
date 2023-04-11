from enum import Enum


class DataTypes(Enum):
    """Defines numerical types of each column."""
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(Enum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index
