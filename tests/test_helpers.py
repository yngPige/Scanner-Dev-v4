import pandas as pd
from src.utils.helpers import calculate_technical_indicators

def test_calculate_technical_indicators():
    df = pd.DataFrame({
        'open': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'high': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        'low': [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9],
        'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'volume': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })
    result = calculate_technical_indicators(df)
    # Check that some indicators are present
    assert 'RSI_14' in result.columns
    assert 'MACD_12_26_9' in result.columns
    assert 'EMA_20' in result.columns
    # assert 'SMA_20' in result.columns
    assert 'BBL_20_2.0' in result.columns
    assert 'ATR_14' in result.columns
    assert 'STOCHk_14_3_3' in result.columns
    # Check that output shape matches input
    assert result.shape[0] == df.shape[0] 