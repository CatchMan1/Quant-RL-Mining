from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
import os

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    AMOUNT = 6
    TWAP = 7
    TOP_AMOUNT = 8
    SUCCESSIVE_INCREASE_AMPLITUDE = 9
    SUCCESSIVE_DECREASE_AMPLITUDE = 10
    SUCCESSIVE_INCREASE_AMOUNT = 11
    SUCCESSIVE_DECREASE_AMOUNT = 12
    SUCCESSIVE_INCREASE_COUNT = 13
    SUCCESSIVE_DECREASE_COUNT = 14
    AMPLITUDE_MAX = 15
    AMPLITUDE_MEAN = 16
    AMPLITUDE_STD = 17
    TOP_AMOUNT_RET = 18
    MAX_RET = 19
    INCREASE_RET_COUNT = 20
    DECREASE_RET_COUNT = 21
    EQUAL_RET_COUNT = 22
    MIN_RET = 23
    MAX_RET_INDEX = 24
    MIN_RET_INDEX = 25
    MAX_AMOUNT_INDEX = 26
    INCREASE_KLINE_RATIO = 27
    DECREASE_KLINE_RATIO = 28
    EQUAL_KLINE_RATIO = 29
    VWAP_EMA_1 = 30
    VWAP_EMA_2 = 31

# Map FeatureType to CSV column names
_FEATURE_TO_COL = {
    FeatureType.OPEN: 'open',
    FeatureType.CLOSE: 'close',
    FeatureType.HIGH: 'high',
    FeatureType.LOW: 'low',
    FeatureType.VOLUME: 'vol',
    FeatureType.VWAP: 'vwap',
    FeatureType.AMOUNT: 'amount',
    FeatureType.TWAP: 'twap',
    FeatureType.TOP_AMOUNT: 'top_amount',
    FeatureType.SUCCESSIVE_INCREASE_AMPLITUDE: 'successive_increase_amplitude',
    FeatureType.SUCCESSIVE_DECREASE_AMPLITUDE: 'successive_decrease_amplitude',
    FeatureType.SUCCESSIVE_INCREASE_AMOUNT: 'successive_increase_amount',
    FeatureType.SUCCESSIVE_DECREASE_AMOUNT: 'successive_decrease_amount',
    FeatureType.SUCCESSIVE_INCREASE_COUNT: 'successive_increase_count',
    FeatureType.SUCCESSIVE_DECREASE_COUNT: 'successive_decrease_count',
    FeatureType.AMPLITUDE_MAX: 'amplitude_max',
    FeatureType.AMPLITUDE_MEAN: 'amplitude_mean',
    FeatureType.AMPLITUDE_STD: 'amplitude_std',
    FeatureType.TOP_AMOUNT_RET: 'top_amount_ret',
    FeatureType.MAX_RET: 'max_ret',
    FeatureType.INCREASE_RET_COUNT: 'increase_ret_count',
    FeatureType.DECREASE_RET_COUNT: 'decrease_ret_count',
    FeatureType.EQUAL_RET_COUNT: 'equal_ret_count',
    FeatureType.MIN_RET: 'min_ret',
    FeatureType.MAX_RET_INDEX: 'max_ret_index',
    FeatureType.MIN_RET_INDEX: 'min_ret_index',
    FeatureType.MAX_AMOUNT_INDEX: 'max_amount_index',
    FeatureType.INCREASE_KLINE_RATIO: 'increase_kline_ratio',
    FeatureType.DECREASE_KLINE_RATIO: 'decrease_kline_ratio',
    FeatureType.EQUAL_KLINE_RATIO: 'equal_kline_ratio',
    FeatureType.VWAP_EMA_1: 'vwap_ema_1',
    FeatureType.VWAP_EMA_2: 'vwap_ema_2',
}
class StockData:
    def __init__(self,
                 csv_path: str = 'data/kline_im.csv',
                 start_time: str = '2022-10-10',
                 end_time: str = '2024-12-31',
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0'),
                 ) -> None:
        self.csv_path = csv_path
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._timestamps = self._get_data()

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index]:
        df = pd.read_csv(self.csv_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime').reset_index(drop=True)

        # Filter by time range
        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)
        mask = (df['DateTime'] >= start_ts) & (df['DateTime'] <= end_ts)
        df = df[mask].reset_index(drop=True)

        timestamps = df['DateTime']

        # Extract features: shape (n_bars, n_features)
        feature_cols = [_FEATURE_TO_COL[f] for f in self._features]
        values = df[feature_cols].values.astype(np.float32)

        # Keep 3D shape (n_bars, n_features, 1) for backward compatibility with expression system
        values = values.reshape(values.shape[0], len(self._features), 1) # 单股票情况
        return torch.tensor(values, dtype=torch.float, device=self.device), timestamps

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return 1

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.max_future_days == 0:
            date_index = self._timestamps.iloc[self.max_backtrack_days:]
        else:
            date_index = self._timestamps.iloc[self.max_backtrack_days:-self.max_future_days]
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=date_index, columns=columns)
