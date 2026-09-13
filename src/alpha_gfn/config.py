from alphagen.data.expression import *
from alphagen_qlib.stock_data import FeatureType

# GFN Task Hyperparameters
MAX_EXPR_LENGTH = 30

# GFN Model Hyperparameters
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Action Space
OPERATORS = [
    # Unary
    Abs, SLog1p, Inv, Sign, Log,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
    TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA,
    # Pair rolling
    TsCov, TsCorr
]

FEATURES = [
    FeatureType.OPEN,
    FeatureType.CLOSE,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.VWAP,
    FeatureType.AMOUNT,
    FeatureType.TWAP,
    FeatureType.TOP_AMOUNT,
    FeatureType.SUCCESSIVE_INCREASE_AMPLITUDE,
    FeatureType.SUCCESSIVE_DECREASE_AMPLITUDE,
    FeatureType.SUCCESSIVE_INCREASE_AMOUNT,
    FeatureType.SUCCESSIVE_DECREASE_AMOUNT,
    FeatureType.SUCCESSIVE_INCREASE_COUNT,
    FeatureType.SUCCESSIVE_DECREASE_COUNT,
    FeatureType.AMPLITUDE_MAX,
    FeatureType.AMPLITUDE_MEAN,
    FeatureType.AMPLITUDE_STD,
    FeatureType.TOP_AMOUNT_RET,
    FeatureType.MAX_RET,
    FeatureType.INCREASE_RET_COUNT,
    FeatureType.DECREASE_RET_COUNT,
    FeatureType.EQUAL_RET_COUNT,
    FeatureType.MIN_RET,
    FeatureType.MAX_RET_INDEX,
    FeatureType.MIN_RET_INDEX,
    FeatureType.MAX_AMOUNT_INDEX,
    FeatureType.INCREASE_KLINE_RATIO,
    FeatureType.DECREASE_KLINE_RATIO,
    FeatureType.EQUAL_KLINE_RATIO,
    FeatureType.VWAP_EMA_1,
    FeatureType.VWAP_EMA_2,
]

DELTA_TIMES = [4, 8, 16, 32, 64]
