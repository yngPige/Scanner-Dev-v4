import pandas as pd
import pandas_ta as ta
from typing import Any, List, Optional
import os
import logging

# Import per-exchange fetchers
from Data.okx_public_data import fetch_okx_ohlcv
from Data.blofin_oublic_data import BlofinWebSocketClient, fetch_blofin_ohlcv
from Data.mexc_public_data import fetch_mexc_ohlcv
from Data.ccxt_public_data import fetch_ccxt_ohlcv


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
    df.ta.ema(close='close', length=9, append=True)
    df.ta.ema(close='close', length=20, append=True)
    df.ta.ema(close='close', length=50, append=True)
    df.ta.ema(close='close', length=100, append=True)
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

def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = '1m',
    limit: int = 100,
    use_ws: bool = False,
    ws_timeout: float = 5.0
) -> Optional[List[List[Any]]]:
    """
    Generic OHLCV fetcher for multiple exchanges. Dispatches to the correct per-exchange function.

    Args:
        exchange (str): Exchange name (e.g., 'OKX', 'Blofin', 'MEXC').
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): Timeframe string (e.g., '1m', '1H', etc.).
        limit (int): Number of bars to fetch.
        use_ws (bool): For exchanges supporting WebSocket (e.g., Blofin), use WS if True.
        ws_timeout (float): Timeout for WebSocket fetch (if used).

    Returns:
        Optional[List[List[Any]]]: List of OHLCV rows or None if error.

    Raises:
        ValueError: If the exchange is not supported.

    Example:
        >>> fetch_ohlcv('OKX', 'BTC/USDT', '1H', 500)
        >>> fetch_ohlcv('Blofin', 'BTC/USDT', '1H', 500, use_ws=True)
        >>> fetch_ohlcv('MEXC', 'BTC/USDT', '1H', 500)

    To add a new exchange:
        1. Implement fetch_<exchange>_ohlcv in Data/<exchange>_public_data.py.
        2. Import and add a branch below.
        3. Ensure the function returns List[List[Any]]: [timestamp, open, high, low, close, volume].
    """
    exchange = exchange.strip().lower()
    try:
        if exchange == 'okx':
            return fetch_okx_ohlcv(symbol, timeframe, limit)
        elif exchange == 'blofin':
            if use_ws:
                inst_id = symbol.replace('/', '-') + '-SPOT' if not symbol.endswith('-SPOT') else symbol
                ws_client = BlofinWebSocketClient()
                return ws_client.fetch_ohlcv(inst_id, bar=timeframe, limit=limit, timeout=ws_timeout)
            else:
                inst_id = symbol.replace('/', '-') + '-SPOT' if not symbol.endswith('-SPOT') else symbol
                return fetch_blofin_ohlcv(inst_id, bar=timeframe, limit=limit)
        elif exchange == 'mexc':
            return fetch_mexc_ohlcv(symbol, timeframe, limit)
        elif exchange.startswith('ccxt:'):
            ccxt_exch = exchange.split(':', 1)[1]
            return fetch_ccxt_ohlcv(ccxt_exch, symbol, timeframe, limit)
        else:
            raise ValueError(f"Exchange '{exchange}' not supported in fetch_ohlcv.")
    except Exception as exc:
        logging.error(f"Error fetching OHLCV for {exchange} {symbol} {timeframe}: {exc}")
        return None

