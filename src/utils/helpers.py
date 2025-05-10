import pandas as pd
import pandas_ta as ta
from typing import Any
import os


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and append common technical indicators to a price DataFrame using pandas-ta.

    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        pd.DataFrame: DataFrame with technical indicators as additional columns.

    Raises:
        ValueError: If required columns are missing or data is insufficient.

    Example:
        >>> df = calculate_technical_indicators(df)
    """
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Ensure all required columns are numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs in required columns
    df = df.dropna(subset=list(required_cols))
    if df.empty:
        raise ValueError("Input DataFrame has no valid rows after dropping NaNs in required columns.")

    # Check for minimum rows for indicators (MACD needs at least 26, use 35 for safety)
    MIN_ROWS = 35
    if len(df) < MIN_ROWS:
        raise ValueError(f"Not enough data for indicator calculation. At least {MIN_ROWS} rows required, got {len(df)}.")

    # Check for all-NaN or empty close column
    if df['close'].isnull().all() or df['close'].empty:
        raise ValueError("The 'close' column is empty or all NaN after cleaning.")

    print("DataFrame shape before indicators:", df.shape)
    print("DataFrame columns:", df.columns)
    print("First few rows:\n", df.head())
    print("Any NaNs in required columns?", df[['open', 'high', 'low', 'close', 'volume']].isnull().any())
    print("Number of rows:", len(df))

    # Calculate indicators
    df = df.copy()
    df.ta.rsi(close='close', length=14, append=True)
    df.ta.macd(close='close', append=True)
    df.ta.ema(close='close', length=21, append=True)
    df.ta.sma(close='close', length=20, append=True)
    df.ta.bbands(close='close', length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.stoch(high='high', low='low', k=14, d=3, append=True)
    df.ta.adx(high='high', low='low', close='close', length=14, append=True)
    df.ta.obv(close='close', volume='volume', append=True)
    df.ta.willr(high='high', low='low', close='close', length=14, append=True)
    # Add more indicators as needed

    df = df.dropna(axis=1, how='all')
    df = df.ffill().bfill()
    return df 

def compute_support_levels(df: pd.DataFrame, window: int = 20) -> list[float]:
    """
    Compute support levels as local minima over a rolling window.
    Args:
        df (pd.DataFrame): DataFrame with 'low' column
        window (int): Rolling window size
    Returns:
        List[float]: List of support levels
    """
    if 'low' not in df.columns:
        return []
    lows = df['low']
    supports = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    # Filter to only those that are also minima in the window
    supports = supports[ (supports.rolling(window, center=True).min() == supports) ]
    return sorted(supports.dropna().unique().tolist())

def compute_resistance_levels(df: pd.DataFrame, window: int = 20) -> list[float]:
    """
    Compute resistance levels as local maxima over a rolling window.
    Args:
        df (pd.DataFrame): DataFrame with 'high' column
        window (int): Rolling window size
    Returns:
        List[float]: List of resistance levels
    """
    if 'high' not in df.columns:
        return []
    highs = df['high']
    resistances = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    resistances = resistances[ (resistances.rolling(window, center=True).max() == resistances) ]
    return sorted(resistances.dropna().unique().tolist()) 

