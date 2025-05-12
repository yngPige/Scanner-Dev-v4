import logging
from typing import Any, Dict, List, Optional
import requests
import os
from dotenv import load_dotenv


load_dotenv()  # Loads variables from .env into os.environ

api_key = os.environ.get("BLOFIN_API_KEY")
secret = os.environ.get("BLOFIN_SECRET")
passphrase = os.environ.get("BLOFIN_PASSPHRASE")

logger = logging.getLogger(__name__)


def test_api() -> Dict[str, Any]:
    """Test OKX and Blofin API connectivity and basic public endpoint functionality, including Blofin WebSocket.

    Returns:
        Dict[str, Any]: Dictionary with status and details of the API test.

    Example:
        >>> test_api()
    """
    result = {"okx": {"status": "success", "details": {}}, "blofin": {"status": "success", "details": {}}}
    # OKX Test
    try:
        markets_url = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
        markets_resp = requests.get(markets_url, timeout=10)
        markets_resp.raise_for_status()
        markets_data = markets_resp.json()
        result["okx"]["details"]["markets_count"] = len(markets_data.get('data', []))
        ticker = fetch_okx_ticker('BTC/USDT')
        result["okx"]["details"]["ticker"] = ticker.get('lastPx') if ticker else None
    except Exception as exc:
        logger.error(f"OKX API test failed: {exc}")
        result["okx"]["status"] = "error"
        result["okx"]["details"]["error"] = str(exc)
    # Blofin Test
    try:
        from Data.blofin_oublic_data import fetch_blofin_instruments, fetch_blofin_ticker, BlofinWebSocketClient
        api_key = os.environ.get("BLOFIN_API_KEY")
        secret = os.environ.get("BLOFIN_SECRET")
        passphrase = os.environ.get("BLOFIN_PASSPHRASE")
        instruments = fetch_blofin_instruments(api_key, secret, passphrase)
        result["blofin"]["details"]["markets_count"] = len(instruments) if instruments else 0
        # If fetch_blofin_ticker requires auth, pass credentials
        try:
            ticker = fetch_blofin_ticker(api_key, secret, passphrase, 'BTC-USDT-SPOT')
        except TypeError:
            ticker = fetch_blofin_ticker('BTC-USDT-SPOT')
        result["blofin"]["details"]["ticker"] = ticker.get('last') if ticker else None
        # Test Blofin WebSocket OHLCV
        ws_client = BlofinWebSocketClient()
        ws_ohlcv = ws_client.fetch_ohlcv('BTC-USDT-SPOT', bar='1m', limit=5, timeout=5.0)
        result["blofin"]["details"]["websocket_ohlcv_sample"] = ws_ohlcv[:2] if ws_ohlcv else []
    except Exception as exc:
        logger.error(f"Blofin API test failed: {exc}")
        result["blofin"]["status"] = "error"
        result["blofin"]["details"]["error"] = str(exc)
    return result


def normalize_pair(symbol: str) -> str:
    """Ensure the symbol is in the correct BASE/USDT format."""
    parts = symbol.split('/')
    if len(parts) == 2:
        return symbol  # Already correct
    elif len(parts) > 2:
        # Remove duplicate quote parts
        return f"{parts[0]}/USDT"
    else:
        # If only base is given, append USDT
        return f"{symbol}/USDT"


def fetch_okx_ticker(symbol: str, all_markets: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch ticker data for a symbol or all tickers from OKX public API."""
    try:
        if all_markets:
            url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get('data', [])
        else:
            if not symbol:
                return None
            symbol = normalize_pair(symbol)
            inst_id = symbol.replace("/", "-")
            url = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get('data'):
                return data['data'][0]
            return None
    except Exception as exc:
        logging.error(f"OKX error fetching ticker: {exc}")
        return None


def fetch_okx_ohlcv(symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[List[List[Any]]]:
    """Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol from OKX public API."""
    VALID_BARS = {'1m', '3m', '5m', '15m', '30m', '1H', '4H', '1D', '1W', '1M'}
    if timeframe not in VALID_BARS:
        logger.error(f"Invalid timeframe '{timeframe}'. Must be one of {VALID_BARS}")
        return None
    try:
        if not symbol:
            logger.error("No symbol provided to fetch_okx_ohlcv.")
            return None
        symbol = normalize_pair(symbol)
        inst_id = symbol.replace("/", "-")
        url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={timeframe}&limit={limit}"
        logger.info(f"Requesting OKX OHLCV: {url}")
        resp = requests.get(url, timeout=10)
        logger.info(f"Response status: {resp.status_code}")
        try:
            data = resp.json()
        except Exception as json_exc:
            logger.error(f"Failed to parse JSON from OKX response: {json_exc}")
            logger.error(f"Raw response text: {resp.text}")
            return None
        if data.get('data'):
            ohlcv = []
            for row in data['data']:
                ts = int(row[0])
                o, h, l, c, vol = map(float, row[1:6])
                ohlcv.append([ts, o, h, l, c, vol])
            logger.info(f"Fetched {len(ohlcv)} OHLCV rows for {symbol}.")
            return ohlcv
        else:
            logger.error(f"OKX returned no data for {symbol}: {data}")
        return None
    except Exception as exc:
        logger.error(f"OKX error fetching OHLCV: {exc}", exc_info=True)
        return None



if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    # Example usage
    print("OKX API Test:")
    pprint.pprint(test_api())
    print("\nOKX Ticker:")
    pprint.pprint(fetch_okx_ticker('BTC/USDT'))
    print("\nOKX OHLCV:")
    pprint.pprint(fetch_okx_ohlcv('BTC/USDT', timeframe='1m', limit=5)) 