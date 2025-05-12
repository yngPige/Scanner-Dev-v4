import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict

def ma_crossover_strategy(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> Dict[str, pd.Series]:
    """
    Simple Moving Average Crossover strategy.
    Entry: Buy when fast MA crosses above slow MA, Sell when fast MA crosses below slow MA.

    Args:
        df (pd.DataFrame): OHLCV data with 'close' column.
        fast (int): Fast MA period.
        slow (int): Slow MA period.

    Returns:
        Dict[str, pd.Series]: 'entry' and 'exit' boolean Series.
    """
    df = df.copy()
    df['fast_ma'] = df['close'].rolling(fast).mean()
    df['slow_ma'] = df['close'].rolling(slow).mean()
    df['signal'] = 0
    df.loc[fast:, 'signal'] = np.where(df['fast_ma'][fast:] > df['slow_ma'][fast:], 1, -1)
    df['entry'] = (df['signal'].shift(1) < 0) & (df['signal'] > 0)
    df['exit'] = (df['signal'].shift(1) > 0) & (df['signal'] < 0)
    return {'entry': df['entry'], 'exit': df['exit']}

def rsi_strategy(df: pd.DataFrame, period: int = 14, overbought: float = 70, oversold: float = 30) -> Dict[str, pd.Series]:
    """
    RSI Overbought/Oversold strategy.
    Entry: Buy when RSI crosses above oversold, Sell when RSI crosses below overbought.

    Args:
        df (pd.DataFrame): OHLCV data with 'close' column.
        period (int): RSI period.
        overbought (float): Overbought threshold.
        oversold (float): Oversold threshold.

    Returns:
        Dict[str, pd.Series]: 'entry' and 'exit' boolean Series.
    """
    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=period)
    df['entry'] = (df['rsi'].shift(1) < oversold) & (df['rsi'] >= oversold)
    df['exit'] = (df['rsi'].shift(1) > overbought) & (df['rsi'] <= overbought)
    return {'entry': df['entry'], 'exit': df['exit']}

def macd_strategy(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    MACD Crossover strategy.
    Entry: Buy when MACD crosses above signal, Sell when MACD crosses below signal.

    Args:
        df (pd.DataFrame): OHLCV data with 'close' column.

    Returns:
        Dict[str, pd.Series]: 'entry' and 'exit' boolean Series.
    """
    df = df.copy()
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['entry'] = (df['macd'].shift(1) < df['macd_signal'].shift(1)) & (df['macd'] > df['macd_signal'])
    df['exit'] = (df['macd'].shift(1) > df['macd_signal'].shift(1)) & (df['macd'] < df['macd_signal'])
    return {'entry': df['entry'], 'exit': df['exit']} 