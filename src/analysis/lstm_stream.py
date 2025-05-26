import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
import time

class LSTMDataStreamer(QThread):
    prediction_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, fetch_func, lstm_model, window_size: int, interval: int = 60):
        """
        Args:
            fetch_func: Callable that returns the latest OHLCV DataFrame.
            lstm_model: Trained LSTMRegressor instance.
            window_size: Number of bars for LSTM input.
            interval: Fetch interval in seconds.
        """
        super().__init__()
        self.fetch_func = fetch_func
        self.lstm_model = lstm_model
        self.window_size = window_size
        self.interval = interval
        self._running = True

    def run(self):
        while self._running:
            try:
                df = self.fetch_func()
                if df is not None and len(df) >= self.window_size:
                    feature_cols = [c for c in df.columns if c not in ['timestamp', 'close']]
                    data = df[feature_cols].values[-self.window_size:]
                    X = data.reshape(1, self.window_size, -1)
                    forecast, uncertainty = self.lstm_model.predict_with_uncertainty(X, n_iter=20)
                    last_close = df['close'].iloc[-1]
                    delta = forecast - last_close
                    percent_change = (delta / last_close) * 100 if last_close != 0 else 0.0
                    result = {
                        'forecast': float(forecast),
                        'uncertainty': float(uncertainty),
                        'delta': float(delta),
                        'percent_change': float(percent_change),
                        'last_close': float(last_close),
                        'timestamp': df['timestamp'].iloc[-1]
                    }
                    self.prediction_ready.emit(result)
            except Exception as e:
                self.error_occurred.emit(str(e))
            time.sleep(self.interval)

    def stop(self):
        self._running = False 