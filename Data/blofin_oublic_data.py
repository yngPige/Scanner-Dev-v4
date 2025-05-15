import logging
from typing import Any, Dict, List, Optional, Set
import requests
import hmac
import hashlib
import base64
import time
import uuid
import json
import threading
import websocket

logger = logging.getLogger(__name__)

BLOFIN_BASE_URL = "https://openapi.blofin.com"
BLOFIN_WS_URL = "wss://openapi.blofin.com/ws/public"

def generate_blofin_signature(secret: str, method: str, path: str, timestamp: str, nonce: str, body: str = "") -> str:
    """
    Generate Blofin API signature.
    """
    prehash = f"{path}{method}{timestamp}{nonce}{body}"
    hex_signature = hmac.new(secret.encode(), prehash.encode(), hashlib.sha256).hexdigest().encode()
    return base64.b64encode(hex_signature).decode()

def  fetch_blofin_instruments(api_key: str, secret: str, passphrase: str, inst_type: str = "SPOT") -> Optional[List[Dict[str, Any]]]:
    """
    Fetch all available trading instruments (symbols) from Blofin with authentication.

    Args:
        api_key (str): Blofin API key.
        secret (str): Blofin API secret.
        passphrase (str): Blofin API passphrase.
        inst_type (str): Instrument type, e.g., "SPOT", "SWAP", "FUTURES", "MARGIN".

    Returns:
        Optional[List[Dict[str, Any]]]: List of instrument dicts, or None on error.
    """
    path = f"/api/v1/public/instruments?instType={inst_type}"
    url = f"{BLOFIN_BASE_URL}{path}"
    method = "GET"
    timestamp = str(int(time.time() * 1000))
    nonce = str(uuid.uuid4())
    body = ""
    signature = generate_blofin_signature(secret, method, path, timestamp, nonce, body)
    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-NONCE": nonce,
        "ACCESS-PASSPHRASE": passphrase,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Blofin instruments raw response: {data}")
        return data.get("data", [])
    except Exception as exc:
        logger.error(f"Blofin error fetching instruments: {exc}")
        return None

def fetch_blofin_ticker(inst_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch ticker data for a given instrument ID from Blofin.

    Args:
        inst_id (str): Instrument ID, e.g., "BTC-USDT-SPOT".

    Returns:
        Optional[Dict[str, Any]]: Ticker data dict, or None on error.
    """
    url = f"{BLOFIN_BASE_URL}/api/v1/public/ticker"
    params = {"instId": inst_id}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            return data["data"][0]
        return None
    except Exception as exc:
        logger.error(f"Blofin error fetching ticker: {exc}")
        return None

def fetch_blofin_ohlcv(inst_id: str, bar: str = "1m", limit: int = 100) -> Optional[List[List[Any]]]:
    """
    Fetch OHLCV (candlestick) data for a given instrument ID from Blofin.

    Args:
        inst_id (str): Instrument ID, e.g., "BTC-USDT-SPOT".
        bar (str): Timeframe, e.g., "1m", "5m", "1h", "1d".
        limit (int): Number of data points to fetch.

    Returns:
        Optional[List[List[Any]]]: List of [timestamp, open, high, low, close, volume], or None on error.
    """
    url = f"{BLOFIN_BASE_URL}/api/v1/public/candlesticks"
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            # Each row: [ts, o, h, l, c, vol, ...]
            ohlcv = []
            for row in data["data"]:
                ts = int(row[0])
                o, h, l, c, vol = map(float, row[1:6])
                ohlcv.append([ts, o, h, l, c, vol])
            return ohlcv
        return None
    except Exception as exc:
        logger.error(f"Blofin error fetching OHLCV: {exc}")
        return None

class BlofinWebSocketClient:
    """
    WebSocket client for Blofin public market data (instruments, OHLCV, etc).
    """
    def __init__(self):
        self.ws_url = BLOFIN_WS_URL
        self.ws = None
        self.last_ohlcv = []
        self._running = False

    def _on_message(self, ws, message):
        data = json.loads(message)
        if 'arg' in data and data['arg'].get('channel') == 'candlesticks':
            # Data: {'arg': {...}, 'data': [[ts, o, h, l, c, vol, ...]]}
            self.last_ohlcv = data.get('data', [])
            logger.info(f"Received OHLCV: {self.last_ohlcv}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
        self._running = False

    def _on_open(self, ws):
        logger.info("WebSocket connection opened.")

    def fetch_ohlcv(self, inst_id: str, bar: str = "1m", limit: int = 100, timeout: float = 10.0) -> List[List[Any]]:
        """
        Fetch OHLCV data for a given instrument via WebSocket. Returns the most recent candles.
        """
        self.last_ohlcv = []
        self._running = True
        def run():
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=lambda ws: ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{
                        "channel": "candlesticks",
                        "instId": inst_id,
                        "bar": bar
                    }]
                })),
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self.ws = ws
            ws.run_forever()
        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        # Wait for data or timeout
        waited = 0
        while waited < timeout:
            if self.last_ohlcv:
                break
            time.sleep(0.1)
            waited += 0.1
        self._running = False
        if self.ws:
            self.ws.close()
        return self.last_ohlcv

def get_blofin_spot_pairs_via_ws(timeout: float = 10.0) -> Set[str]:
    """
    Fetch all Blofin spot pairs via WebSocket (public instruments channel).
    Returns a set of pairs like 'BTC/USDT'.
    """
    pairs = set()
    done = threading.Event()
    def on_message(ws, message):
        data = json.loads(message)
        if 'arg' in data and data['arg'].get('channel') == 'instruments':
            for inst in data.get('data', []):
                if inst.get('quoteCcy', '') == 'USDT':
                    base = inst.get('baseCcy')
                    quote = inst.get('quoteCcy')
                    if base and quote:
                        pairs.add(f"{base}/USDT")
            done.set()
            ws.close()
    def on_open(ws):
        ws.send(json.dumps({
            "op": "subscribe",
            "args": [{
                "channel": "instruments",
                "instType": "SPOT"
            }]
        }))
    ws = websocket.WebSocketApp(
        BLOFIN_WS_URL,
        on_open=on_open,
        on_message=on_message
    )
    t = threading.Thread(target=ws.run_forever)
    t.daemon = True
    t.start()
    done.wait(timeout)
    return pairs

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    print("Blofin Instruments (SPOT):")
    pprint.pprint(fetch_blofin_instruments("api_key", "secret", "passphrase"))
    print("\nBlofin Ticker (BTC-USDT-SPOT):")
    pprint.pprint(fetch_blofin_ticker("BTC-USDT-SPOT"))
    print("\nBlofin OHLCV (BTC-USDT-SPOT, 1m):")
    pprint.pprint(fetch_blofin_ohlcv("BTC-USDT-SPOT", bar="1m", limit=5)) 