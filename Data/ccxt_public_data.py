import ccxt
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_ccxt_ohlcv(
    exchange_name: str,
    symbol: str,
    timeframe: str = '1m',
    limit: int = 100,
    **kwargs
) -> Optional[List[List[Any]]]:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol from any CCXT-supported exchange.

    Args:
        exchange_name (str): Name of the exchange (e.g., 'binance', 'okx', 'mexc').
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', etc.).
        limit (int): Number of bars to fetch.
        **kwargs: Additional keyword arguments for the exchange constructor.

    Returns:
        Optional[List[List[Any]]]: List of OHLCV rows or None if error.

    Example:
        >>> fetch_ccxt_ohlcv('binance', 'BTC/USDT', '1h', 100)
    """
    try:
        exchange_class = getattr(ccxt, exchange_name.lower())
        exchange = exchange_class({"enableRateLimit": True, **kwargs})
        if not exchange.has.get('fetchOHLCV', False):
            logger.error(f"Exchange {exchange_name} does not support fetchOHLCV.")
            return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return ohlcv
    except Exception as exc:
        logger.error(f"CCXT error fetching OHLCV for {exchange_name} {symbol} {timeframe}: {exc}")
        return None


def fetch_ccxt_ticker(
    exchange_name: str,
    symbol: str,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Fetch ticker data for a symbol from any CCXT-supported exchange.

    Args:
        exchange_name (str): Name of the exchange (e.g., 'binance', 'okx', 'mexc').
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        **kwargs: Additional keyword arguments for the exchange constructor.

    Returns:
        Optional[Dict[str, Any]]: Ticker data dictionary or None if error.

    Example:
        >>> fetch_ccxt_ticker('binance', 'BTC/USDT')
    """
    try:
        exchange_class = getattr(ccxt, exchange_name.lower())
        exchange = exchange_class({"enableRateLimit": True, **kwargs})
        if not exchange.has.get('fetchTicker', False):
            logger.error(f"Exchange {exchange_name} does not support fetchTicker.")
            return None
        ticker = exchange.fetch_ticker(symbol)
        return ticker
    except Exception as exc:
        logger.error(f"CCXT error fetching ticker for {exchange_name} {symbol}: {exc}")
        return None


def fetch_ccxt_ohlcv_all_pairs(
    exchange_name: str,
    timeframe: str = '1m',
    limit: int = 100,
    quote: str = 'USDT',
    **kwargs
) -> Dict[str, Optional[List[List[Any]]]]:
    """
    Fetch OHLCV data for all pairs on a CCXT exchange for a given timeframe and quote currency.

    Args:
        exchange_name (str): Name of the exchange (e.g., 'binance').
        timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', etc.).
        limit (int): Number of bars to fetch per pair.
        quote (str): Quote currency to filter pairs (default 'USDT').
        **kwargs: Additional keyword arguments for the exchange constructor.

    Returns:
        Dict[str, Optional[List[List[Any]]]]: Mapping from symbol to OHLCV data or None if error.

    Example:
        >>> fetch_ccxt_ohlcv_all_pairs('binance', '1h', 10)
    """
    import time
    results = {}
    try:
        exchange_class = getattr(ccxt, exchange_name.lower())
        exchange = exchange_class({"enableRateLimit": True, **kwargs})
        if not exchange.has.get('fetchOHLCV', False):
            logger.error(f"Exchange {exchange_name} does not support fetchOHLCV.")
            return {}
        # Validate timeframe
        valid_timeframes = getattr(exchange, 'timeframes', None)
        if valid_timeframes is not None and timeframe not in valid_timeframes:
            logger.error(f"Timeframe '{timeframe}' not supported by {exchange_name}. Supported: {list(valid_timeframes.keys())}")
            return {}
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith(f'/{quote}') and markets[s].get('active', True)]
        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                results[symbol] = ohlcv
                time.sleep(exchange.rateLimit / 1000.0)  # Respect rate limit
            except Exception as exc:
                logger.error(f"Error fetching OHLCV for {symbol}: {exc}")
                results[symbol] = None
        return results
    except Exception as exc:
        logger.error(f"CCXT error fetching all pairs OHLCV for {exchange_name}: {exc}")
        return {}


if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    print("Binance BTC/USDT 1m OHLCV:")
    pprint.pprint(fetch_ccxt_ohlcv('binance', 'BTC/USDT', '1m', 5))
    print("\nBinance BTC/USDT Ticker:")
    pprint.pprint(fetch_ccxt_ticker('binance', 'BTC/USDT'))
    print("\nBinance ALL USDT pairs 1m OHLCV (first 2 shown):")
    all_ohlcv = fetch_ccxt_ohlcv_all_pairs('binance', '1m', 2)
    for k, v in list(all_ohlcv.items())[:2]:
        print(f"{k}: {v}") 