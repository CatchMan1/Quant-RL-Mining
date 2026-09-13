import sys
import os
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'src'))

import torch
import numpy as np
import argparse
import json
import pandas as pd
from typing import List

from alphagen.data.expression import *
from alphagen.data.expression import OutOfDataRangeError
from alphagen_qlib.stock_data import StockData, FeatureType


def _make_expr_eval_namespace() -> dict:
    import alphagen.data.expression as _expr
    ns = {}
    for name in dir(_expr):
        if not name.startswith('_') and name[0].isupper():
            ns[name] = getattr(_expr, name)

    feature_names = [
        'close', 'open', 'high', 'low', 'volume', 'vwap',
        'amount', 'twap', 'top_amount',
        'successive_increase_amplitude', 'successive_decrease_amplitude',
        'successive_increase_amount', 'successive_decrease_amount',
        'successive_increase_count', 'successive_decrease_count',
        'amplitude_max', 'amplitude_mean', 'amplitude_std',
        'top_amount_ret', 'max_ret', 'increase_ret_count',
        'decrease_ret_count', 'equal_ret_count', 'min_ret',
        'max_ret_index', 'min_ret_index', 'max_amount_index',
        'increase_kline_ratio', 'decrease_kline_ratio', 'equal_kline_ratio',
        'vwap_ema_1', 'vwap_ema_2',
    ]
    for name in feature_names:
        ns[name] = Feature(getattr(FeatureType, name.upper()))
    return ns


def load_expressions(pool_path: str) -> List[Expression]:
    ns = _make_expr_eval_namespace()
    with open(pool_path, encoding='utf-8') as f:
        raw = json.load(f)
    exprs_raw = raw['exprs']
    exprs = []
    for expr_raw in exprs_raw:
        clean = expr_raw.replace('$', '')
        exprs.append(eval(clean, {"__builtins__": {}}, ns))
    return exprs


def main():
    parser = argparse.ArgumentParser(description='Export factor values from a pool JSON to CSV')
    parser.add_argument('--pool', type=str, default='./data/gfn_logs/pool_10/test/pool.json',
                        help='Path to the pool JSON file (e.g. data/gfn_logs/pool_xxx/pool_999.json)')
    parser.add_argument('--csv', type=str, default='data/kline_im.csv',
                        help='Path to the kline CSV data file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: pool_path with .factors.csv suffix)')
    parser.add_argument('--start', type=str, default='2022-10-10',
                        help='Start date for data loading')
    parser.add_argument('--end', type=str, default='2026-03-09',
                        help='End date for data loading')
    parser.add_argument('--max-backtrack', type=int, default=200,
                        help='Max backtrack days for rolling operators')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    print(f"[Pool] Loading {args.pool}")
    exprs = load_expressions(args.pool)
    print(f"[Pool] {len(exprs)} factors loaded")

    print(f"[Data] Loading {args.csv} [{args.start} ~ {args.end}]")
    stock_data = StockData(
        csv_path=args.csv,
        start_time=args.start,
        end_time=args.end,
        max_backtrack_days=args.max_backtrack,
        max_future_days=0,
        device=device
    )

    timestamps = stock_data._timestamps
    n_bars = len(timestamps)
    valid_len = stock_data.n_days
    valid_timestamps = timestamps.iloc[args.max_backtrack : args.max_backtrack + valid_len]
    print(f"[Data] {n_bars} bars loaded, valid: {valid_len} rows ({valid_timestamps.iloc[0]} ~ {valid_timestamps.iloc[-1]})")

    factor_values = {}
    failed = 0
    n_valid = len(valid_timestamps)
    for i, expr in enumerate(exprs):
        try:
            value = expr.evaluate(stock_data)
            value = value.squeeze().cpu().numpy()
            factor_values[f'factor_{i}'] = value
            print(f"[Factor {i}] OK  {str(expr)[:80]}")
        except (OutOfDataRangeError, RuntimeError):
            factor_values[f'factor_{i}'] = np.full(n_valid, np.nan)
            failed += 1
            print(f"[Factor {i}] SKIP (OutOfDataRange)  {str(expr)[:80]}")

    print(f"[Data] Reading pct column from {args.csv}")
    raw_df = pd.read_csv(args.csv)
    raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'])
    raw_df = raw_df.set_index('DateTime').sort_index()
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    raw_df = raw_df[(raw_df.index >= start_ts) & (raw_df.index <= end_ts)]
    pct_series = raw_df['pct']

    df = pd.DataFrame(factor_values, index=valid_timestamps)
    df.index.name = 'DateTime'
    df['pct'] = pct_series.reindex(valid_timestamps).values

    if args.output is None:
        args.output = args.pool.replace('.json', '_factors.csv')

    df.to_csv(args.output)
    print(f"[Done] Saved {df.shape[1]} factors x {df.shape[0]} rows to {args.output}")
    if failed > 0:
        print(f"[Warn] {failed} factors failed (OutOfDataRange), filled with NaN")


if __name__ == '__main__':
    main()
