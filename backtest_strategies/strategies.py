import numpy as np
import pandas as pd
from typing import Any

def moving_average_crossover_strategy(df: pd.DataFrame, short: int = 10, long: int = 30) -> np.ndarray:
    """
    Moving Average Crossover strategy.
    Buy (1) when short MA crosses above long MA, Sell (0) otherwise.

    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.
        short (int): Short window length.
        long (int): Long window length.

    Returns:
        np.ndarray: Array of signals (1 for buy, 0 for sell).
    """
    short_ma = df['close'].rolling(window=short).mean()
    long_ma = df['close'].rolling(window=long).mean()
    signal = (short_ma > long_ma).astype(int)
    return signal.values

def rsi_strategy(df: pd.DataFrame, lower: float = 30, upper: float = 70, period: int = 14) -> np.ndarray:
    """
    RSI Overbought/Oversold strategy.
    Buy (1) when RSI < lower, Sell (0) when RSI > upper, Hold (-1) otherwise.

    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.
        lower (float): Oversold threshold.
        upper (float): Overbought threshold.
        period (int): RSI period.

    Returns:
        np.ndarray: Array of signals (1 for buy, 0 for sell, -1 for hold).
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signal = np.full(len(df), -1)
    signal[rsi < lower] = 1
    signal[rsi > upper] = 0
    return signal

def buy_and_hold_strategy(df: pd.DataFrame, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    Buy & Hold strategy: always buy at the start, hold thereafter.

    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.

    Returns:
        np.ndarray: Array of signals (1 for buy, -1 for hold).
    """
    signal = np.full(len(df), -1)
    if len(df) > 0:
        signal[0] = 1
    return signal 