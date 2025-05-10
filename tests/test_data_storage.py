import os
import pandas as pd
import pytest
from src.utils.data_storage import save_ohlcv_to_csv, load_ohlcv_from_csv

def test_save_and_load(tmp_path):
    # Create dummy OHLCV DataFrame
    df = pd.DataFrame({
        'timestamp': [1, 2, 3],
        'open': [1.0, 2.0, 3.0],
        'high': [1.1, 2.1, 3.1],
        'low': [0.9, 1.9, 2.9],
        'close': [1.05, 2.05, 3.05],
        'volume': [10, 20, 30]
    })
    file_path = tmp_path / 'test.csv'
    save_ohlcv_to_csv(df, str(file_path))
    loaded = load_ohlcv_from_csv(str(file_path))
    pd.testing.assert_frame_equal(df, loaded)

def test_merge(tmp_path):
    df1 = pd.DataFrame({'timestamp': [1, 2], 'open': [1, 2], 'high': [1, 2], 'low': [1, 2], 'close': [1, 2], 'volume': [1, 2]})
    df2 = pd.DataFrame({'timestamp': [2, 3], 'open': [2, 3], 'high': [2, 3], 'low': [2, 3], 'close': [2, 3], 'volume': [2, 3]})
    file_path = tmp_path / 'merge.csv'
    save_ohlcv_to_csv(df1, str(file_path))
    save_ohlcv_to_csv(df2, str(file_path))
    loaded = load_ohlcv_from_csv(str(file_path))
    assert set(loaded['timestamp']) == {1, 2, 3} 