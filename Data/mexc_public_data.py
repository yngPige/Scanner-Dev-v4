import logging
from typing import Any, Dict, List, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into os.environ

logger = logging.getLogger(__name__)

# MEXC valid timeframes (bars)
VALID_BARS = {'1m', '5m', '15m', '30m', '1H', '4H', '1D', '1W', '1M'}

# MEXC API base URL
MEXC_BASE_URL = "https://api.mexc.com"


def normalize_pair(symbol: str) -> str:
    """
    Ensure the symbol is in the correct BASEUSDT format for MEXC.
    Args:
        symbol (str): Symbol in BASE/USDT or BASEUSDT format.
    Returns:
        str: Normalized symbol for MEXC (e.g., BTCUSDT)
    """
    if "/" in symbol:
        base, quote = symbol.split("/")
        return f"{base}{quote}"
    return symbol


def normalize_timeframe(timeframe: str) -> str:
    """
    Normalize timeframe to MEXC supported format.
    Args:
        timeframe (str): Timeframe string (e.g., '1h', '1H', '1d', etc.)
    Returns:
        str: Normalized timeframe (e.g., '1m', '1H', '1D', etc.)
    """
    tf_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1H', '1H': '1H', '4h': '4H', '4H': '4H',
        '1d': '1D', '1D': '1D', '1w': '1W', '1W': '1W', '1M': '1M'
    }
    return tf_map.get(timeframe, timeframe)


def fetch_mexc_ticker(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch ticker data for a symbol from MEXC public API.
    Args:
        symbol (str): Symbol in BASE/USDT or BASEUSDT format.
    Returns:
        Optional[Dict[str, Any]]: Ticker data dictionary or None if error.
    """
    try:
        symbol = normalize_pair(symbol)
        url = f"{MEXC_BASE_URL}/api/v3/ticker/24hr?symbol={symbol}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as exc:
        logger.error(f"MEXC error fetching ticker: {exc}")
        return None


def fetch_mexc_ohlcv(symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[List[List[Any]]]:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol from MEXC public API.
    Args:
        symbol (str): Symbol in BASE/USDT or BASEUSDT format.
        timeframe (str): Timeframe string (e.g., '1m', '1H', etc.)
        limit (int): Number of bars to fetch (max 1000 for MEXC)
    Returns:
        Optional[List[List[Any]]]: List of OHLCV lists or None if error.
    """
    tf = normalize_timeframe(timeframe)
    if tf not in VALID_BARS:
        logger.error(f"Invalid timeframe '{timeframe}'. Must be one of {VALID_BARS}")
        return None
    try:
        symbol = normalize_pair(symbol)
        url = f"{MEXC_BASE_URL}/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Defensive: Ensure data is a list (MEXC returns list of lists on success, dict on error)
        if not isinstance(data, list):
            logger.error(f"MEXC OHLCV API returned non-list: {data}")
            return None
        # MEXC returns: [ [open_time, open, high, low, close, volume, ...], ... ]
        ohlcv = [row[:6] for row in data]
        return ohlcv
    except Exception as exc:
        logger.error(f"MEXC error fetching OHLCV: {exc}")
        return None 