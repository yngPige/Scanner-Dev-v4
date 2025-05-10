import os
import pandas as pd
from typing import List, Any

def get_historical_path(symbol: str, timeframe: str) -> str:
    """
    Get the file path for storing historical OHLCV data for a given symbol and timeframe.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC-USDT')
        timeframe (str): Timeframe string (e.g., '1h', '15m')

    Returns:
        str: Path to the CSV file for this symbol and timeframe.
    """
    safe_symbol = symbol.replace("/", "-")
    return os.path.join("Data", "historical", f"{safe_symbol}_{timeframe}.csv")

def merge_ohlcv_rows(row1: pd.Series, row2: pd.Series) -> pd.Series:
    """
    Merge two OHLCV rows with the same timestamp, preferring non-null values and the maximum for numeric columns.
    """
    merged = {}
    for col in row1.index:
        v1, v2 = row1[col], row2[col]
        if pd.isna(v1) and not pd.isna(v2):
            merged[col] = v2
        elif not pd.isna(v1) and pd.isna(v2):
            merged[col] = v1
        elif not pd.isna(v1) and not pd.isna(v2):
            # For numeric columns, take the maximum (except for volume, which should be summed)
            if col == "volume":
                merged[col] = v1 + v2
            elif col in ["open", "high", "low", "close"]:
                merged[col] = max(v1, v2)
            else:
                merged[col] = v1  # For timestamp or unknown columns
        else:
            merged[col] = None
    return pd.Series(merged)

def save_ohlcv(symbol: str, timeframe: str, ohlcv: List[List[Any]]) -> None:
    """
    Save OHLCV data to a CSV file, appending and merging by timestamp with advanced deduplication.

    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe string
        ohlcv (List[List[Any]]): List of OHLCV rows
    """
    os.makedirs(os.path.join("Data", "historical"), exist_ok=True)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = get_historical_path(symbol, timeframe)
    if os.path.exists(path):
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        # Group by timestamp and merge rows
        merged = combined.groupby("timestamp", as_index=False).agg(lambda x: x.dropna().max() if x.name != "volume" else x.dropna().sum())
        # For each group, if there are multiple rows, merge them using merge_ohlcv_rows
        # (This is a fallback for any non-numeric columns or if more complex logic is needed)
        # merged = merged.apply(lambda group: group if len(group) == 1 else merge_ohlcv_rows(group.iloc[0], group.iloc[1]), axis=1)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged.to_csv(path, index=False)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(path, index=False)

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical OHLCV data from CSV for a given symbol and timeframe.

    Args:
        symbol (str): Trading pair symbol
        timeframe (str): Timeframe string

    Returns:
        pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    path = get_historical_path(symbol, timeframe)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]) 