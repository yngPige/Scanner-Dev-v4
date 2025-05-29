import sys
import os
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
import time
import importlib.util
from typing import Any, List, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QAbstractItemView, QDoubleSpinBox, QSpinBox, QInputDialog, QMessageBox, QProgressBar, QCompleter, QDialog, QListWidget, QDialogButtonBox, QCheckBox, QTabWidget, QScrollArea, QToolButton, QProgressDialog, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QIcon
from datetime import datetime, timezone, timedelta
from Data.okx_public_data import fetch_okx_ticker
from backtest_strategies.strategies import moving_average_crossover_strategy, rsi_strategy, buy_and_hold_strategy
from src.utils.data_storage import save_ohlcv, load_ohlcv
from src.utils.helpers import fetch_ohlcv
import pandas as pd
import pandas_ta as ta
import numpy as np
from dotenv import load_dotenv
import os
import requests
from src.analysis.ml import MLAnalyzer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import plotly.graph_objects as go
from PyQt6.QtWebEngineWidgets import QWebEngineView  # Requires PyQtWebEngine
import tempfile
from Data.ccxt_public_data import fetch_ccxt_ohlcv
import joblib
from plotly.subplots import make_subplots
from src.analysis.lstm_stream import LSTMDataStreamer
import psutil
import pyqtgraph as pg

load_dotenv()

class MainWindow(QMainWindow):
    timeframe_map = {
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1H',
        '1H': '1H',
        '4h': '4H',
        '4H': '4H',
        '1d': '1D',
        '1D': '1D',
        '1w': '1W',
        '1W': '1W',
        '1m': '1m',
        '3m': '3m',
        '1M': '1M',
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3lacks Trading Terminal")
        self.setGeometry(100, 100, 900, 700)
        self.selected_algorithm = "RandomForest"  # Default
        self.ml_analyzer = MLAnalyzer()
        self.pair_sources = {}  # Ensure pair_sources is always initialized
        self.ccxt_pair_cache = {}  # Added for caching CCXT pairs
        # --- Ensure ccxt_exchanges is always initialized ---
        self.ccxt_exchanges = ["binanceus", "okx", "kraken", "bybit", "kucoin"]
        # --- Initialize all attributes that may be referenced before assignment ---
        self.llm_output = None
        self.analysis_label = None
        self.pair_menu_widget = None
        self.chart_window = None
        self.chart_view = None
        self.optuna_worker = None
        self.last_df_hist = None
        self.last_analysis = None
        # --- End attribute initialization ---
        self.init_ui()

    def init_ui(self):
        # Layouts
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        llm_layout = QHBoxLayout()

        # Replace OKX Pair search bar with a button
        self.select_pair_button = QPushButton("Select Pair")
        self.select_pair_button.clicked.connect(self.show_pair_menu)
        controls_layout.addWidget(self.select_pair_button)

        # Timeframe selection
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["5m", "15m", "30m", "1h", "4h", "1d", "30d"])
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.timeframe_combo)

        # ML Algorithm Selector
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "RandomForest", "GradientBoosting", "SVM", "XGBoost", "Ensemble (Voting)", "Ensemble (Stacking)", "LSTM"
        ])
        self.algorithm_combo.setCurrentText(self.selected_algorithm)
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        controls_layout.addWidget(QLabel("ML Algorithm:"))
        controls_layout.addWidget(self.algorithm_combo)

        # Analysis Mode Selector
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["Machine Learning", "LLM", "Both"])
        controls_layout.addWidget(QLabel("Analysis Mode:"))
        controls_layout.addWidget(self.analysis_mode_combo)

        # Ollama Model Selector (local models only)
        self.ollama_model_combo = QComboBox()
        local_ollama_models = []
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if 'models' in data:
                    local_ollama_models = [m['name'] for m in data['models'] if 'name' in m]
        except Exception:
            pass
        if not local_ollama_models:
            local_ollama_models = ["llama2"]  # fallback
        self.ollama_model_combo.addItems(local_ollama_models)
        self.ollama_model_combo.setCurrentText(local_ollama_models[0])
        controls_layout.addWidget(QLabel("Ollama Model:"))
        controls_layout.addWidget(self.ollama_model_combo)

        # Fetch and Analyze buttons
        self.fetch_button = QPushButton("Fetch Data")
        self.fetch_button.clicked.connect(self.fetch_data)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.feature_report_button = QPushButton("Feature Report")
        self.feature_report_button.clicked.connect(self.show_feature_report_window)
        # Add Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_training_and_analysis)

        # Data Table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(7)
        self.data_table.setHorizontalHeaderLabels([
            "Timestamp", "Open", "High", "Low", "Close", "Volume", "Change %"
        ])
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setVisible(False)

        # Strategy Selector
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "ML Model",
            "Moving Average Crossover",
            "RSI",
            "Buy & Hold"
        ])
        controls_layout.addWidget(QLabel("Backtest Strategy:"))
        controls_layout.addWidget(self.strategy_combo)

        # Backtest Button
        self.backtest_button = QPushButton("Backtest")
        self.backtest_button.clicked.connect(self.run_backtest)

        # Show Chart Toggle Button
        self.show_chart_button = QPushButton("Show Chart")
        self.show_chart_button.setCheckable(True)
        self.show_chart_button.toggled.connect(self.toggle_chart_window)

        # Add Current section for real-time data
        self.current_label = QLabel("<b>Current:</b> --")
        self.current_label.setTextFormat(Qt.TextFormat.RichText)
        # Add reversal indicator label
        self.reversal_label = QLabel("Reversal: --")
        self.reversal_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        # Data Timestamp Label
        self.data_timestamp_label = QLabel("<b>Last Data Fetch:</b> --")
        self.data_timestamp_label.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(self.data_timestamp_label)

        # Add widgets to main_layout (after creation)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.fetch_button)
        main_layout.addWidget(self.data_table)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.train_button)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(self.feature_report_button)
        # Progress Bar (must be initialized before adding to layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.backtest_button)
        main_layout.addWidget(self.show_chart_button)
        main_layout.addWidget(self.current_label)
        main_layout.addWidget(self.reversal_label)  # Add reversal label to layout

        # LSTM Forecast Label
        self.lstm_forecast_label = QLabel("LSTM Forecast: --")
        self.lstm_forecast_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(self.lstm_forecast_label)

        # LSTM Stream Button
        self.lstm_stream_button = QPushButton("Start LSTM Stream")
        self.lstm_stream_button.setCheckable(True)
        self.lstm_stream_button.toggled.connect(self.toggle_lstm_stream)
        main_layout.addWidget(self.lstm_stream_button)

        # LSTM Stream Output Button
        self.lstm_stream_output_button = QPushButton("Show LSTM Stream Output")
        self.lstm_stream_output_button.clicked.connect(self.show_lstm_stream_output_window)
        main_layout.addWidget(self.lstm_stream_output_button)

        # Add Screener button
        self.screener_button = QPushButton("Screener")
        self.screener_button.clicked.connect(self.show_screener_window)
        main_layout.addWidget(self.screener_button)

        # Add Download All Pairs button
        self.download_all_pairs_button = QPushButton("Download All Exchange Pairs (All Timeframes)")
        self.download_all_pairs_button.clicked.connect(self.download_all_pairs_all_timeframes)
        main_layout.addWidget(self.download_all_pairs_button)

        # Set main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # After setting up the UI, check exchange connection
        self.update_exchange_status()

        # Start TickerFetcher thread for real-time updates
        self.ticker_fetcher = TickerFetcher(self.get_current_symbol)
        self.ticker_fetcher.data_fetched.connect(self.update_current_section)
        self.ticker_fetcher.start()

        # --- Auto-select BTC/USDT and 1h timeframe, and fetch data on startup ---
        self.select_pair_button.setText("BTC/USDT")
        self.selected_pair = "BTC/USDT"
        self.selected_exchange = "OKX"
        self.timeframe_combo.setCurrentText("1h")
        self.fetch_data()

        # Timer for auto-refreshing timestamp
        self.data_timestamp_timer = QTimer(self)
        self.data_timestamp_timer.timeout.connect(self.refresh_data_timestamp)
        self.data_timestamp_timer.start(5000)  # 5 seconds
        self.last_data_fetch_time = None

        # --- LSTM Streamer attribute initialization ---
        self.lstm_streamer = None

        # Output window references
        self._lstm_stream_output_dialog = None
        self._lstm_stream_output_table = None

    def update_exchange_status(self):
        """Check if OKX exchange is reachable and update the status label."""
        try:
            resp = requests.get("https://www.okx.com/api/v5/public/time", timeout=3)
            if resp.status_code == 200:
                self.current_label.setText(f"Exchange: OKX | Pair: {self.select_pair_button.text().strip().upper()}")
                self.current_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.current_label.setText(f"Exchange: OKX | Pair: {self.select_pair_button.text().strip().upper()}")
                self.current_label.setStyleSheet("color: red; font-weight: bold;")
        except Exception:
            self.current_label.setText(f"Exchange: OKX | Pair: {self.select_pair_button.text().strip().upper()}")
            self.current_label.setStyleSheet("color: red; font-weight: bold;")

    def get_ccxt_timeframe(self, exchange_name: str, user_timeframe: str) -> str:
        """
        Map the user's selected timeframe to the correct CCXT-supported format for the given exchange.
        """
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class()
            tf_map = getattr(exchange, 'timeframes', {})
            # Try direct match (case-insensitive)
            for tf in tf_map:
                if tf.lower() == user_timeframe.lower():
                    return tf
            # Try normalized (e.g., '1H' -> '1h')
            norm = user_timeframe.lower()
            if norm in tf_map:
                return norm
            # Fallback: return first available
            if tf_map:
                return list(tf_map.keys())[0]
        except Exception as e:
            pass
        return user_timeframe  # fallback to user input if all else fails

    def fetch_data(self):
        self.update_exchange_status()
        # Use the actual selected_pair for CCXT, else construct symbol as before
        if hasattr(self, 'selected_exchange') and self.selected_exchange.lower().startswith("ccxt:"):
            symbol = self.selected_pair
            ccxt_name = self.selected_exchange.split(":", 1)[1]
            norm_timeframe = self.get_ccxt_timeframe(ccxt_name, self.timeframe_combo.currentText())
        else:
            symbol = self.select_pair_button.text().strip().upper()
            norm_timeframe = self.timeframe_map.get(self.timeframe_combo.currentText(), self.timeframe_combo.currentText())
        if not symbol or symbol == "SELECT PAIR":
            self.data_table.setRowCount(0)
            self.data_table.setVisible(False)
            return
        rows = 1000
        exchange = getattr(self, 'selected_exchange', 'OKX')
        use_ws = False
        ohlcv = fetch_ohlcv(exchange, symbol, norm_timeframe, limit=rows, use_ws=use_ws)
        if ohlcv:
            self.data_table.setVisible(True)
            save_ohlcv(symbol, norm_timeframe, ohlcv)
            ohlcv.sort(key=lambda x: x[0], reverse=True)
            top10 = ohlcv[:10]
            self.data_table.setRowCount(len(top10))
            for row_idx, row in enumerate(top10):
                row_data = list(row)
                try:
                    row_data[0] = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
                try:
                    open_price = float(row[1])
                    close_price = float(row[4])
                    bullish = close_price > open_price
                    bearish = close_price < open_price
                    change_pct = ((close_price - open_price) / open_price * 100) if open_price != 0 else 0.0
                except Exception:
                    bullish = False
                    bearish = False
                    change_pct = 0.0
                for col_idx, value in enumerate(row_data):
                    if col_idx in [1, 2, 3, 4]:  # open, high, low, close
                        try:
                            value = f"{float(value):,.4f}"
                        except Exception:
                            pass
                    elif col_idx == 5:
                        try:
                            value = f"{float(value):,.0f}"
                        except Exception:
                            pass
                    item = QTableWidgetItem(str(value))
                    if col_idx == 1:
                        if bullish:
                            item.setForeground(Qt.GlobalColor.green)
                        elif bearish:
                            item.setForeground(Qt.GlobalColor.red)
                    elif col_idx == 4:
                        if bullish:
                            item.setForeground(Qt.GlobalColor.green)
                        elif bearish:
                            item.setForeground(Qt.GlobalColor.red)
                    self.data_table.setItem(row_idx, col_idx, item)
                pct_item = QTableWidgetItem(f"{change_pct:.2f}%")
                self.data_table.setItem(row_idx, 6, pct_item)
            self.last_data_fetch_time = datetime.now(timezone.utc)
            self.refresh_data_timestamp()
            # --- Reversal indicator update ---
            reversal = self.detect_reversal(ohlcv)
            if reversal == 'bullish':
                self.reversal_label.setText("Reversal: Bullish")
                self.reversal_label.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")
            elif reversal == 'bearish':
                self.reversal_label.setText("Reversal: Bearish")
                self.reversal_label.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
            else:
                self.reversal_label.setText("Reversal: --")
                self.reversal_label.setStyleSheet("color: black; font-weight: bold; font-size: 16px;")
        else:
            self.data_table.setRowCount(0)
            self.data_table.setVisible(False)

    def run_analysis(self):
        """Run analysis using MLAnalyzer only."""
        import os
        import joblib
        mode = self.analysis_mode_combo.currentText()
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}"
        else:
            self.show_analysis_window("<b>[Analysis]</b> Please select an exchange and pair.")
            return
        timeframe = self.timeframe_combo.currentText()
        exchange = getattr(self, 'selected_exchange', 'OKX')
        # Normalize timeframe for CCXT
        if str(exchange).lower().startswith("ccxt:"):
            ccxt_name = str(exchange).split(":", 1)[1]
            norm_timeframe = self.get_ccxt_timeframe(ccxt_name, timeframe)
        else:
            norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        model_base = f"model_{symbol.replace('/', '-')}_{norm_timeframe}_{self.selected_algorithm}"
        model_path = os.path.join("Models", model_base + ".pkl")
        model_path_optuna = os.path.join("ModelsOptuna", model_base + ".pkl")
        # Prompt user if both models exist
        use_optuna = False
        if os.path.exists(model_path) and os.path.exists(model_path_optuna):
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Select Model")
            msg.setText("Both regular and hyperparameter-tuned models are available. Which do you want to use for analysis?")
            regular_btn = msg.addButton("Regular", QMessageBox.ButtonRole.AcceptRole)
            optuna_btn = msg.addButton("Hyperparameter-tuned", QMessageBox.ButtonRole.AcceptRole)
            msg.setDefaultButton(optuna_btn)
            msg.exec()
            if msg.clickedButton() == optuna_btn:
                use_optuna = True
        elif os.path.exists(model_path_optuna):
            use_optuna = True
        # Load the selected model if available
        if use_optuna and os.path.exists(model_path_optuna):
            try:
                pipeline = joblib.load(model_path_optuna)
                self.ml_analyzer.pipeline_ = pipeline
            except Exception:
                self.show_analysis_window("<b>[Analysis]</b> Failed to load hyperparameter-tuned model. Using default.")
        elif os.path.exists(model_path):
            try:
                pipeline = joblib.load(model_path)
                self.ml_analyzer.pipeline_ = pipeline
            except Exception:
                self.show_analysis_window("<b>[Analysis]</b> Failed to load regular model. Using default.")
        # Continue with analysis as before
        from src.utils.data_storage import load_ohlcv
        df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
            html = "<b>[Analysis]</b> No historical data available for this symbol/timeframe."
            self.show_analysis_window(html)
            return
        try:
            df_hist = self.add_technical_indicators(df_hist)
        except ValueError as e:
            html = f"<b>[Analysis]</b> {e}"
            self.show_analysis_window(html)
            return
        self.ml_analyzer.set_custom_model_class(None)
        output = ""
        ml_result = None
        if self.ml_analyzer.algorithm != self.selected_algorithm:
            self.on_algorithm_changed(self.selected_algorithm)
        if mode in ["Machine Learning", "Both"]:
            if self.selected_algorithm == "LSTM":
                # Load LSTM model for analysis, auto-train if missing
                try:
                    import numpy as np
                    model_base = f"model_{symbol.replace('/', '-')}_{norm_timeframe}_{self.selected_algorithm}"
                    model_path = os.path.join("Models", model_base + ".pt")
                    from src.analysis.lstm_regressor import LSTMRegressor
                    window = 30  # Use optimal/configurable window size
                    feature_cols = [c for c in df_hist.columns if c not in ['timestamp', 'close']]
                    features = df_hist[feature_cols].values.astype(np.float32)
                    targets = df_hist['close'].values.astype(np.float32)
                    X, y = LSTMRegressor.create_windowed_dataset(features, targets, window)
                    input_size = X.shape[2] if X.ndim == 3 else len(feature_cols)
                    # --- PATCH: Use in-memory model if .pt file is missing ---
                    model_loaded = False
                    if os.path.exists(model_path):
                        model_loaded = True
                    elif hasattr(self.ml_analyzer, 'model') and isinstance(self.ml_analyzer.model, LSTMRegressor):
                        # Check if model is trained (has parameters loaded)
                        model_loaded = True
                    if not model_loaded:
                        log_html = f"<b>[Analysis]</b> LSTM model file not found: {model_path}<br>Please train the LSTM model first using the 'Train Model' button."
                        self.show_analysis_window(log_html, closable=True)
                        return
                    forecast_result = self.ml_analyzer.predict_regression(df_hist)
                    if forecast_result is not None and isinstance(forecast_result, dict):
                        forecast_val = forecast_result.get('forecast', None)
                        if forecast_val is not None:
                            self.lstm_forecast_label.setText(f"LSTM Forecast: {forecast_val:.4f}")
                            output += f"<b>LSTM Next Close Forecast:</b> <span style='color:blue;font-weight:bold'>{forecast_val:.4f}</span><br>"
                        else:
                            self.lstm_forecast_label.setText("LSTM Forecast: --")
                            output += "<b>LSTM Next Close Forecast:</b> <span style='color:red'>Not enough data or model not trained.</span><br>"
                    else:
                        self.lstm_forecast_label.setText("LSTM Forecast: --")
                        output += "<b>LSTM Next Close Forecast:</b> <span style='color:red'>Not enough data or model not trained.</span><br>"
                except Exception as e:
                    self.show_analysis_window(f"<b>[Analysis]</b> Error loading or auto-training LSTM model: {e}")
                    return
            ml_result = self.ml_analyzer.analyze_market_data(
                symbol=symbol,
                klines_data=df_hist,
                additional_context={"timeframe": timeframe}
            )
            output += self.format_ml_analysis_html(ml_result) + "\n\n"
        if mode in ["LLM", "Both"]:
            if not ollama_is_running():
                output += "[Ollama is not running locally on http://localhost:11434. Please start Ollama to use LLM analysis.]\n"
            else:
                ohlcv_data = df_hist[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
                llm_result = self.llm_analyze_ohlcv(ohlcv_data)
                output += "[LLM Analysis]\n" + llm_result
        self.show_analysis_window(output)
        self.last_analysis = ml_result if ml_result and isinstance(ml_result, dict) else None
        self.last_df_hist = df_hist
        # --- Auto-update chart if open and Show Chart is checked ---
        if hasattr(self, 'show_chart_button') and self.show_chart_button.isChecked():
            if not hasattr(self, 'chart_window') or self.chart_window is None:
                self.create_chart_window()
            else:
                self.update_chart_window()

    @staticmethod
    def get_llm_trader_prompt_template() -> str:
        """
        Returns a prompt template for the LLM to act as a technical analyst and provide only technical information about the OHLCV and ML metrics, with no trading recommendations or verbose explanations. Now includes symbol, timeframe, and data range context at the top.
        """
        return """
<b>Symbol:</b> {symbol}<br>
<b>Timeframe:</b> {timeframe}<br>
<b>Data Range:</b> {data_range}<br>

You are a quantitative analyst. Analyze the following OHLCV (Open, High, Low, Close, Volume) data and machine learning (ML) metrics. Provide a concise technical summary of the current market state, including:
- Key technical indicator values (RSI, MACD, moving averages, volatility, etc.)
- Notable support and resistance levels
- Any significant patterns or anomalies in the data

<b>ML Metrics:</b><br>{ml_metrics_html}
<b>OHLCV Data:</b><br><pre>
{ohlcv_csv}
</pre>
"""

    def llm_analyze_ohlcv(self, ohlcv_data):
        # Only use the first 6 columns for OHLCV, for robustness
        ohlcv_data = [row[:6] for row in ohlcv_data]
        # Convert to DataFrame for ML
        df = self.ohlcv_to_df(ohlcv_data)
        # Run ML analysis
        ml_result = self.ml_analyzer.analyze_market_data(
            symbol=self.select_pair_button.text().strip().upper(),
            klines_data=df,
            additional_context={"timeframe": self.timeframe_combo.currentText()}
        )
        # Format ML metrics for LLM prompt
        if isinstance(ml_result, dict) and "error" not in ml_result:
            ml_metrics_html = (
                f"ML Recommendation: {ml_result.get('recommendation')}<br>"
                f"ML Confidence: {ml_result.get('confidence'):.2f}<br>"
                f"ML Direction: {ml_result.get('direction')}<br>"
                f"ML Support Levels: {ml_result.get('support_levels')}<br>"
                f"ML Resistance Levels: {ml_result.get('resistance_levels')}<br>"
                f"ML Key Indicators: {ml_result.get('key_indicators')}<br>"
            )
        else:
            ml_metrics_html = f"ML Analysis Error: {ml_result.get('error', 'Unknown error')}<br>"
        ohlcv_csv = "Timestamp,Open,High,Low,Close,Volume\n" + "\n".join([",".join(map(str, row)) for row in ohlcv_data])
        # Prepare context for the prompt
        symbol = self.select_pair_button.text().strip().upper()
        timeframe = self.timeframe_combo.currentText()
        data_range = f"Last {len(ohlcv_data)} rows"
        prompt_template = self.get_llm_trader_prompt_template()
        prompt = prompt_template.format(
            symbol=symbol,
            timeframe=timeframe,
            data_range=data_range,
            ml_metrics_html=ml_metrics_html,
            ohlcv_csv=ohlcv_csv
        )
        # Use the selected Ollama model from the dropdown
        ollama_model = self.ollama_model_combo.currentText() if hasattr(self, 'ollama_model_combo') else "llama2"
        llm_response = call_ollama_llm(prompt, model=ollama_model, temperature=0.2, max_tokens=512)
        # Display the formatted result in the LLM output box
        self.llm_output.setHtml(f"<b>LLM ({ollama_model}) Technical Summary:</b><br>{llm_response}")
        return llm_response

    def run_backtest(self):
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}/USDT"
        else:
            # Only show error in main area if no symbol
            self.llm_output.setHtml("<b>[Backtest]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
            return
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        rows = 1000
        exchange = getattr(self, 'selected_exchange', 'OKX')
        use_ws = False
        ohlcv = fetch_ohlcv(exchange, symbol, norm_timeframe, limit=rows, use_ws=use_ws)
        if ohlcv:
            save_ohlcv(symbol, norm_timeframe, ohlcv)
            df_hist = load_ohlcv(symbol, norm_timeframe)
        else:
            df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
            # Only show error in main area if no data
            self.llm_output.setHtml("<b>[Backtest]</b> No historical data available for this symbol/timeframe.")
            return
        selected_strategy = self.strategy_combo.currentText()
        if selected_strategy == "ML Model":
            strategy_fn = None
        elif selected_strategy == "Moving Average Crossover":
            strategy_fn = lambda df: moving_average_crossover_strategy(df, short=10, long=30)
        elif selected_strategy == "RSI":
            strategy_fn = rsi_strategy
        elif selected_strategy == "Buy & Hold":
            strategy_fn = buy_and_hold_strategy
        else:
            strategy_fn = None
        if self.ml_analyzer.algorithm != self.selected_algorithm:
            self.on_algorithm_changed(self.selected_algorithm)
        results = self.ml_analyzer.backtest(df_hist, strategy=strategy_fn)
        self.last_backtest_results = results
        # Open a new window to display backtest results
        self.show_backtest_results_window(results, selected_strategy)

    def show_backtest_results_window(self, results: dict, strategy: str):
        """
        Open a new window to display backtest results in a compact, readable way, using a summary table and compact per-split tables. No equity curve chart.
        Each metric has a dropdown for a concise description. Confusion matrix is shown in its own tab per split.
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QTabWidget, QWidget, QHeaderView, QLabel, QHBoxLayout, QMessageBox, QTextEdit
        )
        from PyQt6.QtCore import Qt

        # Concise descriptions for metrics
        metric_descriptions = {
            "accuracy": "Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)",
            "f1": "Harmonic mean of precision and recall (2*P*R/(P+R))",
            "precision": "TP/(TP+FP): Fraction of positive predictions that are correct.",
            "recall": "TP/(TP+FN): Fraction of actual positives that are detected.",
            "sharpe": "Risk-adjusted return (mean return / std dev of return)",
            "cumulative_return": "Total return over the backtest period.",
            "max_drawdown": "Maximum observed loss from a peak to a trough.",
            "balanced_accuracy": "Average of recall obtained on each class.",
            "auc": "Area under the ROC curve.",
            "mean_sharpe": "Average Sharpe ratio across splits.",
            "mean_cumulative_return": "Average cumulative return across splits.",
            "mean_max_drawdown": "Average max drawdown across splits.",
            "algorithm": "The model or strategy used for backtest.",
            "splits": "Number of walk-forward splits.",
            "timestamp": "Time when the backtest was run."
        }

        def metric_info_button(metric):
            btn = QPushButton("?")
            btn.setFixedWidth(22)
            btn.setToolTip("Show description")
            def show_desc():
                desc = metric_descriptions.get(metric, "No description available.")
                QMessageBox.information(btn, metric.replace('_', ' ').capitalize(), desc)
            btn.clicked.connect(show_desc)
            return btn

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Backtest Results - {strategy}")
        dialog.setGeometry(300, 300, 600, 400)
        layout = QVBoxLayout()

        tab_widget = QTabWidget()

        # --- Top-level metrics table (compact, with info buttons) ---
        metrics = [(k, v) for k, v in results.items() if k != "split_metrics" and k != "equity_curve"]
        table = QTableWidget(len(metrics), 3)
        table.setHorizontalHeaderLabels(["Metric", "Value", "Info"])
        for row, (k, v) in enumerate(metrics):
            item = QTableWidgetItem(str(k).replace('_', ' ').capitalize())
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            table.setItem(row, 0, item)
            if isinstance(v, float):
                table.setItem(row, 1, QTableWidgetItem(f"{v:,.4f}"))
            elif isinstance(v, int):
                table.setItem(row, 1, QTableWidgetItem(f"{v:,}"))
            elif isinstance(v, list) and v and all(isinstance(x, (float, int)) for x in v):
                table.setItem(row, 1, QTableWidgetItem(", ".join(f"{x:.4f}" for x in v)))
            else:
                table.setItem(row, 1, QTableWidgetItem(str(v)))
            table.setCellWidget(row, 2, metric_info_button(k))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tab_widget.addTab(table, "Summary")

        # --- Split metrics (compact per-fold, with info buttons) ---
        if "split_metrics" in results and isinstance(results["split_metrics"], list):
            for i, split in enumerate(results["split_metrics"], 1):
                split_tab = QWidget()
                split_layout = QVBoxLayout()
                # Only show key metrics (accuracy, f1, precision, recall, sharpe, cumulative_return, max_drawdown)
                key_metrics = [
                    ("accuracy", split.get("accuracy")),
                    ("f1", split.get("f1")),
                    ("precision", split.get("precision")),
                    ("recall", split.get("recall")),
                    ("sharpe", split.get("sharpe")),
                    ("cumulative_return", split.get("cumulative_return")),
                    ("max_drawdown", split.get("max_drawdown")),
                ]
                key_metrics = [(k, v) for k, v in key_metrics if v is not None]
                split_table = QTableWidget(len(key_metrics), 3)
                split_table.setHorizontalHeaderLabels(["Metric", "Value", "Info"])
                for row, (k, v) in enumerate(key_metrics):
                    item = QTableWidgetItem(str(k).replace('_', ' ').capitalize())
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    split_table.setItem(row, 0, item)
                    if isinstance(v, float):
                        split_table.setItem(row, 1, QTableWidgetItem(f"{v:,.4f}"))
                    elif isinstance(v, int):
                        split_table.setItem(row, 1, QTableWidgetItem(f"{v:,}"))
                    else:
                        split_table.setItem(row, 1, QTableWidgetItem(str(v)))
                    split_table.setCellWidget(row, 2, metric_info_button(k))
                split_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                split_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                split_layout.addWidget(split_table)
                split_tab.setLayout(split_layout)
                tab_widget.addTab(split_tab, f"Split {i}")

                # Confusion matrix tab for this split
                if "confusion_matrix" in split and isinstance(split["confusion_matrix"], list):
                    cm_tab = QWidget()
                    cm_layout = QVBoxLayout()
                    cm_label = QLabel(f"Confusion Matrix (Split {i})")
                    cm_layout.addWidget(cm_label)
                    cm_table = QTableWidget(len(split["confusion_matrix"]), len(split["confusion_matrix"][0]))
                    cm_table.setHorizontalHeaderLabels([str(j) for j in range(len(split["confusion_matrix"][0]))])
                    cm_table.setVerticalHeaderLabels([str(j) for j in range(len(split["confusion_matrix"]))])
                    for r, row_vals in enumerate(split["confusion_matrix"]):
                        for c, val in enumerate(row_vals):
                            cm_table.setItem(r, c, QTableWidgetItem(str(val)))
                    cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                    cm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                    cm_layout.addWidget(cm_table)
                    cm_tab.setLayout(cm_layout)
                    tab_widget.addTab(cm_tab, f"Confusion Matrix (Split {i})")

        layout.addWidget(tab_widget)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec()

    def format_backtest_result_html(self, results: dict, strategy: str) -> str:
        """
        Format backtest results as HTML for display in the GUI.

        Args:
            results (dict): The dictionary of backtest results.
            strategy (str): The name of the strategy used.

        Returns:
            str: HTML-formatted string for display.
        """
        if not results:
            return "<b>[Backtest]</b> No results to display."

        html = f"<h3>Backtest Results: <span style='color:#007acc'>{strategy}</span></h3><ul>"
        for key, value in results.items():
            if key == "split_metrics" and isinstance(value, list):
                html += "<li><b>Split Metrics:</b><ul>"
                for i, split in enumerate(value, 1):
                    html += f"<li><details><summary><b>Split {i}</b></summary><ul>"
                    for k, v in split.items():
                        if k == "confusion_matrix" and isinstance(v, list):
                            html += "<li><b>Confusion Matrix:</b><br><table border='1' style='border-collapse:collapse;'>"
                            for row in v:
                                html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                            html += "</table></li>"
                        elif k == "classification_report" and isinstance(v, dict):
                            html += "<li><b>Classification Report:</b><br><table border='1' style='border-collapse:collapse;'>"
                            # Table header
                            classes = [c for c in v.keys() if isinstance(v[c], dict)]
                            metrics = ["precision", "recall", "f1-score", "support"]
                            html += "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in metrics) + "</tr>"
                            for cls in classes:
                                html += f"<tr><td>{cls}</td>"
                                for m in metrics:
                                    val = v[cls].get(m, "")
                                    if isinstance(val, float):
                                        html += f"<td>{val:.4f}</td>"
                                    else:
                                        html += f"<td>{val}</td>"
                                html += "</tr>"
                            html += "</table></li>"
                        elif isinstance(v, float):
                            html += f"<li><b>{k}:</b> {v:.4f}</li>"
                        elif isinstance(v, int):
                            html += f"<li><b>{k}:</b> {v}</li>"
                        else:
                            html += f"<li><b>{k}:</b> {v}</li>"
                    html += "</ul></details></li>"
                html += "</ul></li>"
            elif isinstance(value, float):
                html += f"<li><b>{key}:</b> {value:,.4f}</li>"
            elif isinstance(value, int):
                html += f"<li><b>{key}:</b> {value:,}</li>"
            elif isinstance(value, dict):
                html += f"<li><b>{key}:</b><ul>"
                for k2, v2 in value.items():
                    html += f"<li>{k2}: {v2}</li>"
                html += "</ul></li>"
            elif isinstance(value, list) and value and all(isinstance(x, (float, int)) for x in value):
                formatted = ', '.join(f"{x:.4f}" for x in value)
                html += f"<li><b>{key}:</b> <span style='font-family:monospace'>{formatted}</span></li>"
            else:
                html += f"<li><b>{key}:</b> {value}</li>"
        html += "</ul>"
        return html

    def show_feature_report_window(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QLabel, QHBoxLayout
        from PyQt6.QtCore import Qt
        import pandas as pd
        import plotly.graph_objects as go
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        import tempfile
        # Use last_df_hist if available, else prompt to fetch/analyze
        df = getattr(self, 'last_df_hist', None)
        if df is None or df.empty:
            self.llm_output.setHtml("<b>[Feature Report]</b> No data available. Please fetch data and run analysis first.")
            return
        # Prepare feature info
        exclude_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_types = [str(df[col].dtype) for col in feature_columns]
        # Sample values (last row)
        sample_row = df.iloc[-1] if not df.empty else None
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Report")
        dialog.setGeometry(350, 350, 900, 600)
        layout = QVBoxLayout()
        label = QLabel("<b>Features Used in Model Training</b>")
        layout.addWidget(label)
        # --- Feature Importances Bar Chart ---
        chart_added = False
        pipeline = getattr(self.ml_analyzer, 'pipeline_', None)
        if pipeline is not None:
            # Try to get feature importances from the model
            try:
                model = pipeline.named_steps.get('model', None)
                if model is not None and hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    # Only use features that are in the current df
                    if len(importances) == len(feature_columns):
                        # Sort by importance descending
                        sorted_idx = importances.argsort()[::-1]
                        sorted_features = [feature_columns[i] for i in sorted_idx]
                        sorted_importances = importances[sorted_idx]
                        fig = go.Figure(go.Bar(
                            x=sorted_importances,
                            y=sorted_features,
                            orientation='h',
                            marker=dict(color='rgba(0,123,255,0.7)')
                        ))
                        fig.update_layout(
                            title="Feature Importances (sorted)",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=120, r=20, t=40, b=40)
                        )
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                            fig.write_html(f.name)
                            chart_path = f.name
                        chart_view = QWebEngineView()
                        from PyQt6.QtCore import QUrl
                        chart_view.load(QUrl.fromLocalFile(chart_path))
                        layout.addWidget(chart_view)
                        chart_added = True
            except Exception as e:
                pass
        # --- Feature Table ---
        table = QTableWidget(len(feature_columns), 3)
        table.setHorizontalHeaderLabels(["Feature Name", "Type", "Sample Value"])
        for i, (col, typ) in enumerate(zip(feature_columns, feature_types)):
            table.setItem(i, 0, QTableWidgetItem(col))
            table.setItem(i, 1, QTableWidgetItem(typ))
            val = sample_row[col] if sample_row is not None and col in sample_row else ""
            table.setItem(i, 2, QTableWidgetItem(str(val)))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(table)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec()

    def on_algorithm_changed(self, algorithm: str):
        """
        Update the selected algorithm, update config, and recreate the MLAnalyzer instance.
        """
        self.selected_algorithm = algorithm
        import config
        config.ML_ALGORITHM = algorithm  # Update global config
        self.ml_analyzer = MLAnalyzer()  # Recreate analyzer with new config

    def download_all_timeframes_for_selected_pair(self, auto=False):
        """
        Download at least a month (30 days) of OHLCV data for all supported timeframes for the selected pair from OKX or CCXT exchanges.
        Skip timeframes if recent data is already downloaded.
        """
        from Data.okx_public_data import fetch_okx_ohlcv
        from Data.ccxt_public_data import fetch_ccxt_ohlcv
        from src.utils.data_storage import save_ohlcv, load_ohlcv
        import time
        from datetime import datetime, timezone, timedelta
        pair = self.select_pair_button.text().strip().upper()
        log_html = ""  # Accumulate log messages here
        if not pair or pair == "SELECT PAIR":
            log_html += "<b>[Download]</b> Please select a pair first.<br>"
            self.show_analysis_window(log_html)
            return
        # Define timeframes for both OKX and CCXT
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        minutes_per_timeframe = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        bars_per_month = {tf: max(30, int((30*24*60)//minutes_per_timeframe[tf])) for tf in minutes_per_timeframe}
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(timeframes))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Downloading a month of all timeframes... %p%")
        QApplication.processEvents()
        # Determine exchange type
        exchange = getattr(self, 'selected_exchange', 'OKX').lower()
        if exchange == "okx":
            fetch_func = fetch_okx_ohlcv
            exchange_label = "OKX"
        elif exchange.startswith("ccxt:"):
            ccxt_name = exchange.split(':', 1)[1]
            def fetch_func(symbol, tf, limit):
                norm_tf = self.get_ccxt_timeframe(ccxt_name, tf)
                return fetch_ccxt_ohlcv(ccxt_name, symbol, norm_tf, limit)
            exchange_label = ccxt_name.upper()
        else:
            log_html += f"<b>[Download]</b> Exchange {exchange} not supported for bulk download.<br>"
            self.progress_bar.setVisible(False)
            self.show_analysis_window(log_html)
            return
        for idx, tf in enumerate(timeframes):
            limit = bars_per_month.get(tf, 30)
            # Check if data is already downloaded and recent
            df_hist = load_ohlcv(pair, tf)
            skip = False
            if not df_hist.empty:
                # Check if last row timestamp is within 24 hours
                last_ts = df_hist.iloc[-1]["timestamp"]
                now = datetime.now(timezone.utc)
                # Handle ms or s timestamps
                if last_ts > 1e10:
                    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                else:
                    last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
                if (now - last_dt) < timedelta(hours=24):
                    skip = True
            if skip:
                log_html += f"<b>[Download]</b> Skipping {pair} ({tf}) [{exchange_label}] - already downloaded and recent.<br>"
                # Smooth progress bar animation for skipped
                current_val = self.progress_bar.value()
                target_val = current_val + 1
                steps = 10
                for step in range(1, steps + 1):
                    self.progress_bar.setValue(current_val + (step * (target_val - current_val)) // steps)
                    QApplication.processEvents()
                    time.sleep(0.01)
                continue
            log_html += f"<b>[Download]</b> Downloading {limit} rows for {pair} ({tf}) from {exchange_label}...<br>"
            QApplication.processEvents()
            ohlcv = fetch_func(pair, tf, limit=limit)
            if ohlcv:
                save_ohlcv(pair, tf, ohlcv)
                log_html += f"<b>[Download]</b> Saved {len(ohlcv)} rows for {pair} ({tf}) [{exchange_label}]<br>"
            else:
                log_html += f"<b>[Download]</b> Failed to fetch data for {pair} ({tf}) [{exchange_label}]<br>"
            # Smooth progress bar animation
            current_val = self.progress_bar.value()
            target_val = current_val + 1
            steps = 10
            for step in range(1, steps + 1):
                self.progress_bar.setValue(current_val + (step * (target_val - current_val)) // steps)
                QApplication.processEvents()
                time.sleep(0.03)
        self.progress_bar.setVisible(False)
        log_html += f"<b>[Download]</b> A month of all timeframes downloaded for {pair} ({exchange_label}).<br>"
        self.show_analysis_window(log_html)

    def toggle_chart_window(self, checked: bool):
        """
        Open or close the chart window when the Show Chart button is toggled.
        If checked, show the chart with the latest data/analysis. If unchecked, close the chart window.
        """
        if checked:
            # If chart_window already exists, just update it
            if hasattr(self, 'chart_window') and self.chart_window is not None:
                self.update_chart_window()
                self.chart_window.show()
            else:
                self.create_chart_window()
        else:
            if hasattr(self, 'chart_window') and self.chart_window:
                self.chart_window.close()
                self.chart_window = None

    def create_chart_window(self):
        """
        Create and show the chart window with the latest data/analysis, including volume, indicator overlays, and subplots.
        """
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtCore import QUrl
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import tempfile
        df = getattr(self, 'last_df_hist', None)
        analysis = getattr(self, 'last_analysis', None)
        if df is None or df.empty:
            return
        # Prepare x-axis
        if 'timestamp' in df.columns:
            x = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].max() > 1e10 else 's')
        else:
            x = df.index
        # Subplots: 1. Candles+indicators, 2. Volume, 3. RSI, 4. MACD
        row_count = 4
        fig = make_subplots(
            rows=row_count, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.55, 0.15, 0.15, 0.15],
            subplot_titles=("Price & Indicators", "Volume", "RSI", "MACD")
        )
        # --- Candlestick ---
        hovertext = [
            f"Time: {xv}<br>Open: {o}<br>High: {h}<br>Low: {l}<br>Close: {c}<br>Volume: {v}" for xv, o, h, l, c, v in zip(x, df['open'], df['high'], df['low'], df['close'], df['volume'])
        ]
        fig.add_trace(go.Candlestick(
            x=x,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Candles',
            increasing_line_color='green', decreasing_line_color='red',
            showlegend=True,
            hovertext=hovertext,
            hoverinfo='text',
        ), row=1, col=1)
        # --- Bollinger Bands ---
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['BBL_20_2.0'],
                mode='lines', name='BB Lower', line=dict(color='rgba(0,0,255,0.3)', dash='dot'),
                hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=df['BBU_20_2.0'],
                mode='lines', name='BB Upper', line=dict(color='rgba(0,0,255,0.3)', dash='dot'),
                hoverinfo='skip'), row=1, col=1)
        # --- EMA overlays ---
        for ma_col, color in [('ema_9', 'blue'), ('ema_20', 'orange'), ('ema_50', 'purple'), ('ema_100', 'brown')]:
            if ma_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=df[ma_col],
                    mode='lines',
                    name=ma_col.upper(),
                    line=dict(color=color, width=1.5, dash='dot'),
                    hoverinfo='skip'), row=1, col=1)
        # --- Trade signal markers ---
        if analysis and isinstance(analysis, dict):
            lo = analysis.get('limit_orders', {})
            for order_type, color, symbol in [
                ('entry', 'green', 'triangle-up'),
                ('take_profit', 'gold', 'star'),
                ('stop_loss', 'red', 'x')
            ]:
                for o in lo.get(order_type, []):
                    price = o.get('price')
                    if price is not None:
                        fig.add_trace(go.Scatter(
                            x=[x.iloc[-1]], y=[price],
                            mode='markers',
                            marker=dict(symbol=symbol, color=color, size=12),
                            name=f"{order_type.title()} @ {price:.4f}",
                            hovertext=f"{order_type.title()}<br>Price: {price:.4f}",
                            hoverinfo='text',
                        ), row=1, col=1)
        # --- Volume ---
        fig.add_trace(go.Bar(
            x=x, y=df['volume'], name='Volume', marker_color='rgba(128,128,128,0.5)',
            hovertext=[f"Volume: {v}" for v in df['volume']], hoverinfo='text'), row=2, col=1)
        # --- RSI ---
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['RSI_14'], mode='lines', name='RSI(14)', line=dict(color='magenta'),
                hovertext=[f"RSI: {r:.2f}" for r in df['RSI_14']], hoverinfo='text'), row=3, col=1)
            fig.add_hline(y=70, line_dash='dot', line_color='red', row=3, col=1)
            fig.add_hline(y=30, line_dash='dot', line_color='green', row=3, col=1)
        # --- MACD ---
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='black'),
                hovertext=[f"MACD: {m:.2f}" for m in df['MACD_12_26_9']], hoverinfo='text'), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=df['MACDs_12_26_9'], mode='lines', name='MACD Signal', line=dict(color='orange', dash='dot'),
                hovertext=[f"Signal: {s:.2f}" for s in df['MACDs_12_26_9']], hoverinfo='text'), row=4, col=1)
            if 'MACDh_12_26_9' in df.columns:
                fig.add_trace(go.Bar(
                    x=x, y=df['MACDh_12_26_9'], name='MACD Hist', marker_color='rgba(0,0,0,0.3)',
                    hovertext=[f"Hist: {h:.2f}" for h in df['MACDh_12_26_9']], hoverinfo='text'), row=4, col=1)
        fig.update_layout(
            title="Candlestick Chart with Analysis",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True,
            template="plotly_white",
            height=900,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        # Save to temp HTML and display in QWebEngineView
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            fig.write_html(f.name)
            chart_path = f.name
        self.chart_window = QWidget()
        self.chart_window.setWindowTitle("Chart View")
        self.chart_window.setGeometry(200, 200, 1000, 900)
        layout = QVBoxLayout()
        self.chart_view = QWebEngineView()
        self.chart_view.load(QUrl.fromLocalFile(chart_path))
        layout.addWidget(self.chart_view)
        self.chart_window.setLayout(layout)
        self.chart_window.show()

    def update_chart_window(self):
        """
        Update the chart in the existing chart window with the latest data/analysis, including volume, indicator overlays, and subplots.
        """
        from PyQt6.QtCore import QUrl
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import tempfile
        df = getattr(self, 'last_df_hist', None)
        analysis = getattr(self, 'last_analysis', None)
        if df is None or df.empty or not hasattr(self, 'chart_view'):
            return
        if 'timestamp' in df.columns:
            x = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].max() > 1e10 else 's')
        else:
            x = df.index
        row_count = 4
        fig = make_subplots(
            rows=row_count, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.55, 0.15, 0.15, 0.15],
            subplot_titles=("Price & Indicators", "Volume", "RSI", "MACD")
        )
        hovertext = [
            f"Time: {xv}<br>Open: {o}<br>High: {h}<br>Low: {l}<br>Close: {c}<br>Volume: {v}" for xv, o, h, l, c, v in zip(x, df['open'], df['high'], df['low'], df['close'], df['volume'])
        ]
        fig.add_trace(go.Candlestick(
            x=x,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Candles',
            increasing_line_color='green', decreasing_line_color='red',
            showlegend=True,
            hovertext=hovertext,
            hoverinfo='text',
        ), row=1, col=1)
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['BBL_20_2.0'],
                mode='lines', name='BB Lower', line=dict(color='rgba(0,0,255,0.3)', dash='dot'),
                hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=df['BBU_20_2.0'],
                mode='lines', name='BB Upper', line=dict(color='rgba(0,0,255,0.3)', dash='dot'),
                hoverinfo='skip'), row=1, col=1)
        for ma_col, color in [('ema_9', 'blue'), ('ema_20', 'orange'), ('ema_50', 'purple'), ('ema_100', 'brown')]:
            if ma_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=df[ma_col],
                    mode='lines',
                    name=ma_col.upper(),
                    line=dict(color=color, width=1.5, dash='dot'),
                    hoverinfo='skip'), row=1, col=1)
        if analysis and isinstance(analysis, dict):
            lo = analysis.get('limit_orders', {})
            for order_type, color, symbol in [
                ('entry', 'green', 'triangle-up'),
                ('take_profit', 'gold', 'star'),
                ('stop_loss', 'red', 'x')
            ]:
                for o in lo.get(order_type, []):
                    price = o.get('price')
                    if price is not None:
                        fig.add_trace(go.Scatter(
                            x=[x.iloc[-1]], y=[price],
                            mode='markers',
                            marker=dict(symbol=symbol, color=color, size=12),
                            name=f"{order_type.title()} @ {price:.4f}",
                            hovertext=f"{order_type.title()}<br>Price: {price:.4f}",
                            hoverinfo='text',
                        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=x, y=df['volume'], name='Volume', marker_color='rgba(128,128,128,0.5)',
            hovertext=[f"Volume: {v}" for v in df['volume']], hoverinfo='text'), row=2, col=1)
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['RSI_14'], mode='lines', name='RSI(14)', line=dict(color='magenta'),
                hovertext=[f"RSI: {r:.2f}" for r in df['RSI_14']], hoverinfo='text'), row=3, col=1)
            fig.add_hline(y=70, line_dash='dot', line_color='red', row=3, col=1)
            fig.add_hline(y=30, line_dash='dot', line_color='green', row=3, col=1)
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='black'),
                hovertext=[f"MACD: {m:.2f}" for m in df['MACD_12_26_9']], hoverinfo='text'), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=df['MACDs_12_26_9'], mode='lines', name='MACD Signal', line=dict(color='orange', dash='dot'),
                hovertext=[f"Signal: {s:.2f}" for s in df['MACDs_12_26_9']], hoverinfo='text'), row=4, col=1)
            if 'MACDh_12_26_9' in df.columns:
                fig.add_trace(go.Bar(
                    x=x, y=df['MACDh_12_26_9'], name='MACD Hist', marker_color='rgba(0,0,0,0.3)',
                    hovertext=[f"Hist: {h:.2f}" for h in df['MACDh_12_26_9']], hoverinfo='text'), row=4, col=1)
        fig.update_layout(
            title="Candlestick Chart with Analysis",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=True,
            template="plotly_white",
            height=900,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            fig.write_html(f.name)
            chart_path = f.name
        self.chart_view.load(QUrl.fromLocalFile(chart_path))

    def get_current_symbol(self) -> str:
        """Return the currently selected trading pair as a string."""
        return self.select_pair_button.text().strip().upper()

    def update_current_section(self, ticker: dict):
        """Update the Current section with real-time data from the thread."""
        if ticker:
            price = ticker.get('last', '--')
            bid = ticker.get('bid', '--')
            ask = ticker.get('ask', '--')
            vol = ticker.get('volCcy24h', '--')
            ts = ticker.get('ts', None)
            if ts:
                ts_str = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = '--'
            html = (
                f"<b>Current:</b> <b>Price:</b> {price} &nbsp; <b>Bid:</b> {bid} &nbsp; <b>Ask:</b> {ask} "
                f"&nbsp; <b>24h Vol:</b> {vol} &nbsp; <b>Time:</b> {ts_str}"
            )
        else:
            html = "<b>Current:</b> --"
        self.current_label.setText(html)

    def refresh_data_timestamp(self):
        """Update the data timestamp label with the last data fetch time."""
        if self.last_data_fetch_time:
            ts_str = self.last_data_fetch_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            self.data_timestamp_label.setText(f"<b>Last Data Fetch:</b> {ts_str}")
        else:
            self.data_timestamp_label.setText("<b>Last Data Fetch:</b> --")

    def detect_reversal(self, ohlcv: list[list[float]]) -> str:
        """
        Detects reversal patterns in OHLCV data using the last 3 candles.
        Supports multiple patterns (bullish/bearish engulfing, hammer, shooting star, etc.).

        Args:
            ohlcv (list[list[float]]): List of OHLCV rows, each as [timestamp, open, high, low, close, volume, ...].

        Returns:
            str: Detected reversal type ('bullish', 'bearish', or 'none').

        Raises:
            ValueError: If ohlcv is not a list of lists or has insufficient data.

        Example:
            >>> candles = [[...], [...], [...]]
            >>> self.detect_reversal(candles)
        """
        if not isinstance(ohlcv, list) or len(ohlcv) < 3:
            return 'none'
        for row in ohlcv[-3:]:
            if not (isinstance(row, list) and len(row) >= 5):
                return 'none'
        prev2, prev1, last = ohlcv[-3], ohlcv[-2], ohlcv[-1]
        patterns = [
            self._is_bullish_engulfing,
            self._is_bearish_engulfing,
            self._is_hammer,
            self._is_shooting_star
        ]
        for pattern_func in patterns:
            result = pattern_func(prev2, prev1, last)
            if result != 'none':
                return result
        return 'none'

    def _is_bullish_engulfing(self, prev2: list[float], prev1: list[float], last: list[float]) -> str:
        """Detect bullish engulfing pattern."""
        # last close > last open, last open < prev1 close, last close > prev1 open
        if last[4] > last[1] and last[1] < prev1[4] and last[4] > prev1[1]:
            return 'bullish'
        return 'none'

    def _is_bearish_engulfing(self, prev2: list[float], prev1: list[float], last: list[float]) -> str:
        """Detect bearish engulfing pattern."""
        # last close < last open, last open > prev1 close, last close < prev1 open
        if last[4] < last[1] and last[1] > prev1[4] and last[4] < prev1[1]:
            return 'bearish'
        return 'none'

    def _is_hammer(self, prev2: list[float], prev1: list[float], last: list[float]) -> str:
        """Detect hammer pattern (bullish reversal)."""
        open_, high, low, close = last[1], last[2], last[3], last[4]
        body = abs(close - open_)
        lower_shadow = open_ - low if open_ > low else close - low
        upper_shadow = high - max(open_, close)
        if body > 0 and lower_shadow > 2 * body and upper_shadow < body:
            return 'bullish'
        return 'none'

    def _is_shooting_star(self, prev2: list[float], prev1: list[float], last: list[float]) -> str:
        """Detect shooting star pattern (bearish reversal)."""
        open_, high, low, close = last[1], last[2], last[3], last[4]
        body = abs(close - open_)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        if body > 0 and upper_shadow > 2 * body and lower_shadow < body:
            return 'bearish'
        return 'none'

    def get_ccxt_pairs(self, exchange_name: str) -> set:
        """
        Fetch and cache the available trading pairs for a given CCXT exchange.

        Args:
            exchange_name (str): The name of the CCXT exchange (e.g., 'binanceus').

        Returns:
            set: A set of trading pair strings (e.g., {'BTC/USDT', ...}).
        """
        if not hasattr(self, 'ccxt_pair_cache'):
            self.ccxt_pair_cache = {}
        if exchange_name in self.ccxt_pair_cache:
            return self.ccxt_pair_cache[exchange_name]
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class()
            markets = exchange.load_markets()
            pairs = set(markets.keys())
            # Only keep USDT pairs for consistency
            usdt_pairs = {p for p in pairs if p.endswith('/USDT')}
            self.ccxt_pair_cache[exchange_name] = usdt_pairs
            return usdt_pairs
        except Exception as e:
            return set([f"Error loading CCXT pairs: {e}"])

    def run_training_and_analysis(self):
        """
        Run training and then analysis in sequence. If Optuna tuning is selected, run analysis only after tuning completes.
        Disable the Run button while running to prevent duplicate runs.
        Before running, ensure all timeframes are downloaded if not already present.
        """
        # Download all timeframes for the selected pair if not already downloaded
        self.download_all_timeframes_for_selected_pair(auto=True)
        self.run_button.setEnabled(False)
        self._run_analysis_after_training = False
        def after_analysis():
            self.run_button.setEnabled(True)
            # Auto-open and update chart if Show Chart is checked
            if hasattr(self, 'show_chart_button') and self.show_chart_button.isChecked():
                if not hasattr(self, 'chart_window') or self.chart_window is None:
                    self.create_chart_window()
                else:
                    self.update_chart_window()
        def after_training():
            self.run_analysis()
            after_analysis()
        # Patch train_model to support callback for Optuna
        def patched_train_model():
            from PyQt6.QtWidgets import QMessageBox, QProgressDialog
            import os
            import joblib
            rows = 1000
            log_html = ""  # Accumulate log messages here
            custom = self.select_pair_button.text().strip().upper()
            if custom:
                symbol = f"{custom}/USDT"
            else:
                log_html += "<b>[ML Training]</b> Please enter a symbol (e.g., BTC) in the Pair field.<br>"
                self.show_analysis_window(log_html)
                after_analysis()
                return
            timeframe = self.timeframe_combo.currentText()
            exchange = getattr(self, 'selected_exchange', 'OKX')
            if str(exchange).lower().startswith("ccxt:"):
                ccxt_name = str(exchange).split(":", 1)[1]
                norm_timeframe = self.get_ccxt_timeframe(ccxt_name, timeframe)
            else:
                norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
            indicator_lookback = max([100, 26, 20, 14])
            min_required_rows = rows + indicator_lookback
            from src.utils.helpers import fetch_ohlcv
            ohlcv = fetch_ohlcv(exchange, symbol, norm_timeframe, limit=min_required_rows)
            from src.utils.data_storage import save_ohlcv, load_ohlcv
            df_hist = None
            if ohlcv:
                save_ohlcv(symbol, norm_timeframe, ohlcv)
                df_hist = load_ohlcv(symbol, norm_timeframe)
            else:
                df_hist = pd.DataFrame()
            if df_hist.empty:
                log_html += "<b>[ML Training]</b> No historical data available for this symbol/timeframe.<br>"
                self.show_analysis_window(log_html)
                after_analysis()
                return
            valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
            if valid_rows < indicator_lookback:
                log_html += (
                    f"<b>[ML Training]</b> Not enough valid data for indicator calculation. At least {indicator_lookback} rows are required after cleaning. Valid rows: {valid_rows}<br>"
                )
                self.show_analysis_window(log_html)
                after_analysis()
                return
            usable_rows = min(rows, valid_rows - indicator_lookback)
            if usable_rows < 1:
                log_html += (
                    f"<b>[ML Training]</b> Not enough data after indicator lookback. Try lowering the rows setting or using a higher timeframe.<br>"
                )
                self.show_analysis_window(log_html)
                after_analysis()
                return
            if usable_rows < rows:
                log_html += (
                    f"<b>[ML Training]</b> Only {usable_rows} rows available for training after indicator calculation. Proceeding with available data.<br>"
                )
            try:
                df_hist = self.add_technical_indicators(df_hist, min_rows=usable_rows)
            except ValueError as e:
                log_html += f"<b>[ML Training]</b> {e}<br>"
                self.show_analysis_window(log_html)
                after_analysis()
                return
            self.ml_analyzer.set_custom_model_class(None)
            model_base = f"model_{symbol.replace('/', '-')}_{norm_timeframe}_{self.selected_algorithm}"
            model_path = os.path.join("Models", model_base + ".pkl")
            model_path_optuna = os.path.join("ModelsOptuna", model_base + ".pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(model_path_optuna), exist_ok=True)
            if "Ensemble" not in self.selected_algorithm:
                reply = QMessageBox.question(self, "Hyperparameter Tuning", "Do you want to run Optuna hyperparameter tuning?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    class OptunaWorker(QThread):
                        result_ready = pyqtSignal(object)
                        error_occurred = pyqtSignal(str)
                        def __init__(self, analyzer, df, model_type):
                            super().__init__()
                            self.analyzer = analyzer
                            self.df = df
                            self.model_type = model_type
                            self._cancel = False
                        def run(self):
                            try:
                                study = self.analyzer.optimize_with_optuna(self.df, model_type=self.model_type, n_trials=20)
                                if self._cancel:
                                    return
                                self.result_ready.emit(study)
                            except Exception as e:
                                self.error_occurred.emit(str(e))
                        def cancel(self):
                            self._cancel = True
                    progress = QProgressDialog("Running Optuna hyperparameter tuning...", "Cancel", 0, 0, self)
                    progress.setWindowTitle("Hyperparameter Tuning")
                    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)
                    progress.setCancelButtonText("Cancel")
                    progress.show()
                    self.optuna_worker = OptunaWorker(self.ml_analyzer, df_hist, self.selected_algorithm)
                    def on_result(study):
                        progress.close()
                        best_trial = study.best_trial
                        best_params = best_trial.params
                        best_score = best_trial.value
                        try:
                            if hasattr(self.ml_analyzer, 'pipeline_'):
                                joblib.dump(self.ml_analyzer.pipeline_, model_path_optuna)
                                log_html_local = log_html + f"<b>[ML Training]</b> Hyperparameter-tuned model saved to {model_path_optuna}<br>"
                            else:
                                log_html_local = log_html + "<b>[ML Training]</b> Warning: No pipeline_ attribute found after Optuna tuning.<br>"
                        except Exception as e:
                            log_html_local = log_html + f"<b>[ML Training]</b> Error saving Optuna-tuned model: {e}<br>"
                        html = f"""
                        <div style='font-family:Arial,sans-serif;'>
                        <h2 style='margin-bottom:0;'>Optuna Hyperparameter Tuning Results</h2>
                        <b>Algorithm:</b> {self.selected_algorithm}<br>
                        <b>Best Score (Accuracy):</b> <span style='color:green;font-weight:bold;'>{best_score:.4f}</span><br>
                        <b>Best Hyperparameters:</b><br>
                        <ul>
                        {''.join(f'<li><b>{k}:</b> {v}</li>' for k, v in best_params.items())}
                        </ul>
                        <b>Timestamp:</b> {datetime.now().isoformat()}<br>
                        <hr style='margin:8px 0;'>
                        </div>
                        """
                        self.show_analysis_window(log_html_local + html)
                        after_training()
                    def on_error(msg):
                        progress.close()
                        self.show_analysis_window(log_html + f"<b>[ML Training]</b> Error during tuning: {msg}<br>")
                        after_analysis()
                    self.optuna_worker.result_ready.connect(on_result)
                    self.optuna_worker.error_occurred.connect(on_error)
                    progress.canceled.connect(self.optuna_worker.cancel)
                    self.optuna_worker.start()
                    return
            # --- Standard training ---
            results = self.ml_analyzer.train_model(df_hist)
            try:
                if hasattr(self.ml_analyzer, 'pipeline_'):
                    joblib.dump(self.ml_analyzer.pipeline_, model_path)
                    log_html += f"<b>[ML Training]</b> Regular model saved to {model_path}<br>"
                else:
                    log_html += "<b>[ML Training]</b> Warning: No pipeline_ attribute found after training.<br>"
            except Exception as e:
                log_html += f"<b>[ML Training]</b> Error saving regular model: {e}<br>"
            log_html += self.format_ml_training_result_html(results)
            self.show_analysis_window(log_html)
            after_training()
        patched_train_model()

    def show_analysis_window(self, html_content: str, closable: bool = True):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis & Recommendations Report")
        dialog.setGeometry(350, 350, 900, 600)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(html_content)
        layout.addWidget(text_edit)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setEnabled(closable)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._analysis_dialog = dialog
        self._analysis_text_edit = text_edit
        self._analysis_close_btn = close_btn

    def update_analysis_window(self, html_content: str, closable: bool = False):
        if hasattr(self, '_analysis_text_edit'):
            self._analysis_text_edit.setHtml(html_content)
        if hasattr(self, '_analysis_close_btn'):
            self._analysis_close_btn.setEnabled(closable)

    def train_model(self):
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import QThread, pyqtSignal
        import os
        import joblib
        rows = 1000  # Default value, or compute based on timeframe/algorithm if needed
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}/USDT"
        else:
            self.show_analysis_window("<b>[ML Training]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
            return
        timeframe = self.timeframe_combo.currentText()
        exchange = getattr(self, 'selected_exchange', 'OKX')
        # Normalize timeframe for CCXT
        if str(exchange).lower().startswith("ccxt:"):
            ccxt_name = str(exchange).split(":", 1)[1]
            norm_timeframe = self.get_ccxt_timeframe(ccxt_name, timeframe)
        else:
            norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        indicator_lookback = max([100, 26, 20, 14])
        min_required_rows = rows + indicator_lookback
        from src.utils.helpers import fetch_ohlcv
        ohlcv = fetch_ohlcv(exchange, symbol, norm_timeframe, limit=min_required_rows)
        from src.utils.data_storage import save_ohlcv, load_ohlcv
        df_hist = None
        if ohlcv:
            save_ohlcv(symbol, norm_timeframe, ohlcv)
            df_hist = load_ohlcv(symbol, norm_timeframe)
        else:
            df_hist = pd.DataFrame()
        if df_hist.empty:
            self.show_analysis_window("<b>[ML Training]</b> No historical data available for this symbol/timeframe.")
            return
        valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
        if valid_rows < indicator_lookback:
            self.show_analysis_window(
                f"<b>[ML Training]</b> Not enough valid data for indicator calculation. At least {indicator_lookback} rows are required after cleaning. Valid rows: {valid_rows}"
            )
            return
        usable_rows = min(rows, valid_rows - indicator_lookback)
        if usable_rows < 1:
            self.show_analysis_window(
                f"<b>[ML Training]</b> Not enough data after indicator lookback. Try lowering the rows setting or using a higher timeframe."
            )
            return
        if usable_rows < rows:
            self.show_analysis_window(
                f"<b>[ML Training]</b> Only {usable_rows} rows available for training after indicator calculation. Proceeding with available data."
            )
        try:
            df_hist = self.add_technical_indicators(df_hist, min_rows=usable_rows)
        except ValueError as e:
            self.show_analysis_window(f"<b>[ML Training]</b> {e}")
            return
        self.ml_analyzer.set_custom_model_class(None)
        # --- Prompt for hyperparameter tuning if supported ---
        model_base = f"model_{symbol.replace('/', '-')}_{norm_timeframe}_{self.selected_algorithm}"
        model_path = os.path.join("Models", model_base + ".pkl")
        model_path_optuna = os.path.join("ModelsOptuna", model_base + ".pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_path_optuna), exist_ok=True)
        if "Ensemble" not in self.selected_algorithm:
            reply = QMessageBox.question(self, "Hyperparameter Tuning", "Do you want to run Optuna hyperparameter tuning?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                class OptunaWorker(QThread):
                    result_ready = pyqtSignal(object)
                    error_occurred = pyqtSignal(str)
                    def __init__(self, analyzer, df, model_type):
                        super().__init__()
                        self.analyzer = analyzer
                        self.df = df
                        self.model_type = model_type
                        self._cancel = False
                    def run(self):
                        try:
                            study = self.analyzer.optimize_with_optuna(self.df, model_type=self.model_type, n_trials=20)
                            if self._cancel:
                                return
                            self.result_ready.emit(study)
                        except Exception as e:
                            self.error_occurred.emit(str(e))
                    def cancel(self):
                        self._cancel = True
                progress = QProgressDialog("Running Optuna hyperparameter tuning...", "Cancel", 0, 0, self)
                progress.setWindowTitle("Hyperparameter Tuning")
                progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.setCancelButtonText("Cancel")
                progress.show()
                self.optuna_worker = OptunaWorker(self.ml_analyzer, df_hist, self.selected_algorithm)
                def on_result(study):
                    progress.close()
                    best_trial = study.best_trial
                    best_params = best_trial.params
                    best_score = best_trial.value
                    # Save the Optuna-tuned model
                    try:
                        if hasattr(self.ml_analyzer, 'pipeline_'):
                            joblib.dump(self.ml_analyzer.pipeline_, model_path_optuna)
                            self.show_analysis_window(f"<b>[ML Training]</b> Hyperparameter-tuned model saved to {model_path_optuna}")
                        else:
                            self.show_analysis_window("<b>[ML Training]</b> Warning: No pipeline_ attribute found after Optuna tuning.")
                    except Exception as e:
                        self.show_analysis_window(f"<b>[ML Training]</b> Error saving Optuna-tuned model: {e}")
                    html = f"""
                    <div style='font-family:Arial,sans-serif;'>
                    <h2 style='margin-bottom:0;'>Optuna Hyperparameter Tuning Results</h2>
                    <b>Algorithm:</b> {self.selected_algorithm}<br>
                    <b>Best Score (Accuracy):</b> <span style='color:green;font-weight:bold;'>{best_score:.4f}</span><br>
                    <b>Best Hyperparameters:</b><br>
                    <ul>
                    {''.join(f'<li><b>{k}:</b> {v}</li>' for k, v in best_params.items())}
                    </ul>
                    <b>Timestamp:</b> {datetime.now().isoformat()}<br>
                    <hr style='margin:8px 0;'>
                    </div>
                    """
                    self.show_analysis_window(html)
                def on_error(msg):
                    progress.close()
                    self.show_analysis_window(f"<b>[ML Training]</b> Error during tuning: {msg}")
                self.optuna_worker.result_ready.connect(on_result)
                self.optuna_worker.error_occurred.connect(on_error)
                progress.canceled.connect(self.optuna_worker.cancel)
                self.optuna_worker.start()
                return
        # --- Standard training ---
        results = self.ml_analyzer.train_model(df_hist)
        # Save the regular model
        try:
            if hasattr(self.ml_analyzer, 'pipeline_'):
                joblib.dump(self.ml_analyzer.pipeline_, model_path)
                self.show_analysis_window(f"<b>[ML Training]</b> Regular model saved to {model_path}")
            else:
                self.show_analysis_window("<b>[ML Training]</b> Warning: No pipeline_ attribute found after training.")
        except Exception as e:
            self.show_analysis_window(f"<b>[ML Training]</b> Error saving regular model: {e}")
        self.show_analysis_window(self.format_ml_training_result_html(results))
        if self.selected_algorithm == "LSTM":
            try:
                from src.analysis.lstm_regressor import LSTMRegressor
                import numpy as np
                window = 30
                feature_cols = [c for c in df_hist.columns if c not in ['timestamp', 'close']]
                features = df_hist[feature_cols].values.astype(np.float32)
                targets = df_hist['close'].values.astype(np.float32)
                X, y = LSTMRegressor.create_windowed_dataset(features, targets, window)
                input_size = X.shape[2] if X.ndim == 3 else len(feature_cols)
                model_base = f"model_{symbol.replace('/', '-')}_{norm_timeframe}_{self.selected_algorithm}"
                model_path = os.path.join("Models", model_base + ".pt")
                log_html = "<b>[ML Training]</b> Preparing to train LSTM..."
                self.show_analysis_window(log_html, closable=False)
                self.lstm_worker = LSTMTrainingWorker(X, y, input_size, model_path)
                self.lstm_worker.progress.connect(lambda html: self.update_analysis_window(html, closable=False))
                def on_finished(model):
                    self.ml_analyzer.model = model
                    self.update_analysis_window(f"<b>[ML Training]</b> LSTM model saved to {model_path}<br>", closable=True)
                self.lstm_worker.finished.connect(on_finished)
                self.lstm_worker.start()
            except Exception as e:
                self.show_analysis_window(f"<b>[ML Training]</b> Error during LSTM training: {e}")
            return
        # Only for non-LSTM models:
        if hasattr(self.ml_analyzer, 'pipeline_'):
            joblib.dump(self.ml_analyzer.pipeline_, model_path)
            self.show_analysis_window(f"<b>[ML Training]</b> Regular model saved to {model_path}")
        # Do not show any window if pipeline_ is missing

    def stop(self):
        self._running = False

    def select_pair_from_menu(self, item):
        label = item.text()
        # Parse pair and exchange(s)
        if "[" in label:
            pair, src = label.split("[")
            pair = pair.strip()
            srcs = src.strip("] ").split(",")
            # Prefer OKX if available, else CCXT
            if "OKX" in srcs:
                self.selected_exchange = "OKX"
            elif any(s.startswith("CCXT:") for s in srcs):
                ccxt_src = next(s for s in srcs if s.startswith("CCXT:"))
                self.selected_exchange = ccxt_src.lower()
            else:
                self.selected_exchange = "OKX"
            self.selected_pair = pair  # Store the actual symbol as shown in CCXT or other exchanges
        else:
            pair = label.strip()
            # If the label is from a CCXT dynamic load, set selected_exchange accordingly
            if "[CCXT:" in label:
                ccxt_src = label.split("[CCXT:")[-1].strip("] ")
                self.selected_exchange = f"ccxt:{ccxt_src.lower()}"
                self.selected_pair = pair  # Store the actual symbol
            else:
                self.selected_exchange = "OKX"
                self.selected_pair = pair
        self.select_pair_button.setText(pair)
        # Remove the menu and restore the data table
        self.centralWidget().layout().removeWidget(self.pair_menu_widget)
        self.pair_menu_widget.deleteLater()
        self.pair_menu_widget = None  # Ensure reference is cleared
        self.data_table.setVisible(True)
        # Automatically fetch data for the selected pair
        self.fetch_data()

    def add_technical_indicators(self, df: pd.DataFrame, min_rows: int = 35) -> pd.DataFrame:
        """
        Add technical indicators to a DataFrame using the shared utility.
        Args:
            df (pd.DataFrame): Input OHLCV DataFrame.
            min_rows (int): Minimum number of rows required.
        Returns:
            pd.DataFrame: DataFrame with indicators.
        Raises:
            ValueError: If not enough data.
        """
        if len(df) < min_rows:
            raise ValueError(f"Not enough data for indicator calculation. At least {min_rows} rows required, got {len(df)}.")
        from src.utils.helpers import calculate_technical_indicators
        return calculate_technical_indicators(df)

    def filter_pair_list(self, text: str = ""):
        """
        Filter the pair list based on the search input and selected exchange.
        """
        search_text = text.lower()
        selected_exchange = self.exchange_combo.currentText()
        self.pair_list.clear()
        for label in self.all_pairs:
            pair = label.split(" [")[0].lower()
            sources = label.lower()
            if search_text in pair:
                if selected_exchange == "All" or selected_exchange.lower() in sources:
                    self.pair_list.addItem(label)

    def format_ml_training_result_html(self, results: dict) -> str:
        """
        Format ML training results as HTML for display in the GUI.

        Args:
            results (dict): The dictionary of training results.

        Returns:
            str: HTML-formatted string for display.
        """
        if not results:
            return "<b>[ML Training]</b> No results to display."
        html = "<h3>ML Training Results</h3><ul>"
        for key, value in results.items():
            if isinstance(value, float):
                html += f"<li><b>{key}:</b> {value:,.4f}</li>"
            elif isinstance(value, int):
                html += f"<li><b>{key}:</b> {value:,}</li>"
            elif isinstance(value, dict):
                html += f"<li><b>{key}:</b><ul>"
                for k2, v2 in value.items():
                    html += f"<li>{k2}: {v2}</li>"
                html += "</ul></li>"
            elif isinstance(value, list) and value and all(isinstance(x, (float, int)) for x in value):
                formatted = ', '.join(f"{x:.4f}" for x in value)
                html += f"<li><b>{key}:</b> <span style='font-family:monospace'>{formatted}</span></li>"
            else:
                html += f"<li><b>{key}:</b> {value}</li>"
        html += "</ul>"
        return html

    def format_ml_analysis_html(self, result: dict) -> str:
        """
        Format the ML analysis result as a visually clear, color-coded HTML summary for display.
        Args:
            result (dict): The dictionary of ML analysis results.
        Returns:
            str: HTML-formatted string for display.
        """
        if not isinstance(result, dict):
            return "<b>[ML Analysis]</b> Invalid result."
        if 'error' in result:
            return f"<b>[ML Analysis Error]</b> {result['error']}"
        # Color helpers
        def color_for_recommendation(rec):
            return 'green' if rec == 'BUY' else 'red' if rec == 'SELL' else 'gray'
        def color_for_direction(dir):
            return 'green' if dir and dir.upper() in ['UP', 'BULLISH'] else 'red' if dir and dir.upper() in ['DOWN', 'BEARISH'] else 'gray'
        def color_for_confidence(val):
            try:
                v = float(val)
                if v >= 80:
                    return 'green'
                elif v >= 60:
                    return 'orange'
                else:
                    return 'red'
            except Exception:
                return 'gray'
        # Start HTML
        html = """
        <div style='font-family:Arial,sans-serif;max-width:700px;'>
        <h2 style='color:#007acc;margin-bottom:0;'>ML Analysis Results</h2>
        <table style='width:100%;margin-bottom:10px;'>
        <tr><td><b>Symbol:</b></td><td>{symbol}</td></tr>
        <tr><td><b>Timeframe:</b></td><td>{timeframe}</td></tr>
        <tr><td><b>Recommendation:</b></td><td style='color:{rec_color};font-weight:bold;'>{recommendation}</td></tr>
        <tr><td><b>Direction:</b></td><td style='color:{dir_color};font-weight:bold;'>{direction}</td></tr>
        <tr><td><b>Confidence:</b></td><td style='color:{conf_color};font-weight:bold;'>{confidence:.4f}</td></tr>
        <tr><td><b>Current Price:</b></td><td>{current_price:,.4f}</td></tr>
        <tr><td><b>Stop Loss:</b></td><td>{stop_loss:,.4f}</td></tr>
        <tr><td><b>Take Profit:</b></td><td>{take_profit:,.4f}</td></tr>
        </table>
        <b>Support Levels:</b> <span style='color:#007acc'>{support_levels}</span><br>
        <b>Resistance Levels:</b> <span style='color:#d2691e'>{resistance_levels}</span><br>
        <b>Top Features:</b> <span style='color:#6a5acd'>{top_features}</span><br>
        <b>Key Indicators:</b> <span style='color:#333'>{key_indicators}</span><br>
        <hr style='margin:10px 0;'>
        <b>Limit Orders:</b><br>{limit_orders_html}
        <hr style='margin:10px 0;'>
        <b>Entry Points:</b> <span style='color:#007acc'>{entry_points}</span><br>
        <b>Exit Points:</b> <span style='color:#228b22'>{exit_points}</span><br>
        <b>Stop Loss Points:</b> <span style='color:#b22222'>{stop_loss_points}</span><br>
        </div>
        """
        # Extract fields with safe fallback
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except Exception:
                return default
        def safe_str(val, default="--"):
            return str(val) if val is not None else default
        symbol = safe_str(result.get('symbol', '--'))
        timeframe = safe_str(result.get('timeframe', '--'))
        recommendation = safe_str(result.get('recommendation', '--'))
        direction = safe_str(result.get('direction', '--'))
        confidence = safe_float(result.get('confidence', 0.0))
        current_price = safe_float(result.get('current_price', 0.0))
        stop_loss = safe_float(result.get('stop_loss', 0.0))
        take_profit = safe_float(result.get('take_profit', 0.0))
        support_levels = ', '.join(f"{safe_float(x):,.4f}" for x in (result.get('support_levels') or []))
        resistance_levels = ', '.join(f"{safe_float(x):,.4f}" for x in (result.get('resistance_levels') or []))
        top_features = ', '.join([safe_str(x) for x in (result.get('top_features') or [])])
        key_indicators = result.get('key_indicators')
        if isinstance(key_indicators, dict):
            key_indicators = ', '.join(f"{k}: {safe_float(v):.2f}" for k, v in key_indicators.items())
        else:
            key_indicators = safe_str(key_indicators)
        # Limit orders (color-coded table)
        def format_limit_orders(lo):
            if not isinstance(lo, dict):
                return str(lo)
            html = "<table style='width:100%;border-collapse:collapse;margin-top:5px;'>"
            html += "<tr style='background:#f0f0f0;'><th>Type</th><th>Price</th><th>Description</th><th>Confidence</th></tr>"
            for k, v in lo.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            conf = item.get('confidence', '--')
                            conf_color = color_for_confidence(conf)
                            html += f"<tr>"
                            html += f"<td style='font-weight:bold;'>{k.title()}</td>"
                            html += f"<td>{item.get('price', '--')}</td>"
                            html += f"<td>{item.get('description', '--')}</td>"
                            html += f"<td style='color:{conf_color};font-weight:bold;'>{conf}</td>"
                            html += "</tr>"
                else:
                    html += f"<tr><td colspan='4'>{v}</td></tr>"
            html += "</table>"
            return html
        limit_orders_html = format_limit_orders(result.get('limit_orders', {}))
        # Points
        entry_points = ', '.join(f"{safe_float(x):,.4f}" for x in (result.get('entry_points') or []))
        exit_points = ', '.join(f"{safe_float(x):,.4f}" for x in (result.get('exit_points') or []))
        stop_loss_points = ', '.join(f"{safe_float(x):,.4f}" for x in (result.get('stop_loss_points') or []))
        # Colors
        rec_color = color_for_recommendation(recommendation)
        dir_color = color_for_direction(direction)
        conf_color = color_for_confidence(confidence)
        # Format
        return html.format(
            symbol=symbol,
            timeframe=timeframe,
            recommendation=recommendation,
            rec_color=rec_color,
            direction=direction,
            dir_color=dir_color,
            confidence=confidence,
            conf_color=conf_color,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            top_features=top_features,
            key_indicators=key_indicators,
            limit_orders_html=limit_orders_html,
            entry_points=entry_points,
            exit_points=exit_points,
            stop_loss_points=stop_loss_points
        )

    def show_pair_menu(self):
        # Toggle: If the menu is already open, close it
        if hasattr(self, 'pair_menu_widget') and self.pair_menu_widget is not None:
            self.centralWidget().layout().removeWidget(self.pair_menu_widget)
            self.pair_menu_widget.deleteLater()
            self.pair_menu_widget = None
            self.data_table.setVisible(True)
            return

        from Data.okx_public_data import fetch_okx_ticker
        # Hide the data table
        self.data_table.setVisible(False)
        # Create the pair menu widget
        self.pair_menu_widget = QWidget()
        layout = QVBoxLayout()
        # Add search bar
        self.pair_search_input = QLineEdit()
        self.pair_search_input.setPlaceholderText("Search pairs...")
        layout.addWidget(self.pair_search_input)
        self.pair_list = QListWidget()
        layout.addWidget(QLabel("Select a Pair:"))
        layout.addWidget(self.pair_list)
        self.ccxt_exchanges = ["binanceus", "okx", "kraken", "bybit", "kucoin"]
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["All", "OKX"] + [f"CCXT:{exch}" for exch in self.ccxt_exchanges])
        layout.addWidget(QLabel("Exchange:"))
        layout.addWidget(self.exchange_combo)
        self.pair_menu_widget.setLayout(layout)
        self.centralWidget().layout().addWidget(self.pair_menu_widget)

        # --- Fetch all pairs for all exchanges (OKX and CCXT) ---
        okx_pairs = set()
        try:
            tickers = fetch_okx_ticker(None, all_markets=True)
            okx_pairs = set(
                t['instId'].replace("-", "/")
                for t in tickers if 'instId' in t and t['instId'].endswith('-USDT')
            )
        except Exception as e:
            okx_pairs = set([f"Error loading OKX pairs: {e}"])
        # Fetch CCXT pairs for all exchanges
        ccxt_pairs_by_exchange = {}
        for exch in self.ccxt_exchanges:
            try:
                pairs = self.get_ccxt_pairs(exch)
                ccxt_pairs_by_exchange[exch] = pairs
            except Exception as e:
                ccxt_pairs_by_exchange[exch] = set([f"Error loading CCXT:{exch} pairs: {e}"])
        # Merge all pairs for 'All'
        all_pairs_set = set(okx_pairs)
        for pairs in ccxt_pairs_by_exchange.values():
            all_pairs_set.update(pairs)
        # Build pair_sources for all pairs
        self.pair_sources = {}
        for p in all_pairs_set:
            srcs = []
            if p in okx_pairs:
                srcs.append("OKX")
            for exch in self.ccxt_exchanges:
                if p in ccxt_pairs_by_exchange[exch]:
                    srcs.append(f"CCXT:{exch}")
            self.pair_sources[p] = srcs
        # Display as: BTC/USDT [OKX,CCXT:binanceus,...]
        self.all_pairs = [f"{p} [{','.join(self.pair_sources[p])}]" for p in sorted(all_pairs_set)]
        self.pair_list.clear()
        self.pair_list.addItems(self.all_pairs)
        self.pair_list.itemClicked.connect(self.select_pair_from_menu)
        self.pair_search_input.textChanged.connect(self.filter_pair_list)
        # --- Dynamic filtering by exchange ---
        def on_exchange_changed():
            selected_exchange = self.exchange_combo.currentText()
            self.pair_list.clear()
            if selected_exchange == "All":
                self.pair_list.addItems(self.all_pairs)
            elif selected_exchange == "OKX":
                okx_labels = [f"{p} [OKX]" for p in sorted(okx_pairs)]
                self.pair_list.addItems(okx_labels)
            elif selected_exchange.startswith("CCXT:"):
                exch = selected_exchange.split(":", 1)[1]
                ccxt_labels = [f"{p} [CCXT:{exch}]" for p in sorted(ccxt_pairs_by_exchange.get(exch, []))]
                self.pair_list.addItems(ccxt_labels)
        self.exchange_combo.currentTextChanged.connect(on_exchange_changed)
        # Initial filter
        self.filter_pair_list(self.pair_search_input.text())

    def start_lstm_stream(self):
        if self.lstm_streamer is not None:
            self.stop_lstm_stream()
        if not hasattr(self.ml_analyzer, 'model') or self.ml_analyzer.model is None:
            self.lstm_forecast_label.setText("LSTM Stream: No model loaded.")
            return
        if not hasattr(self.ml_analyzer, 'lstm_window'):
            self.lstm_forecast_label.setText("LSTM Stream: No LSTM window size.")
            return
        def fetch_latest_ohlcv():
            return getattr(self, 'last_df_hist', None)
        self.lstm_streamer = LSTMDataStreamer(
            fetch_func=fetch_latest_ohlcv,
            lstm_model=self.ml_analyzer.model,
            window_size=getattr(self.ml_analyzer, 'lstm_window', 30),
            interval=60  # seconds
        )
        self.lstm_streamer.prediction_ready.connect(self.handle_lstm_prediction)
        self.lstm_streamer.error_occurred.connect(lambda msg: self.lstm_forecast_label.setText(f"LSTM Stream Error: {msg}"))
        self.lstm_streamer.start()
        self.lstm_forecast_label.setText("LSTM Stream: Running...")

    def stop_lstm_stream(self):
        if self.lstm_streamer is not None:
            self.lstm_streamer.stop()
            self.lstm_streamer.wait()
            self.lstm_streamer = None
            self.lstm_forecast_label.setText("LSTM Stream: Stopped.")

    def toggle_lstm_stream(self, checked):
        if checked:
            self.lstm_stream_button.setText("Stop LSTM Stream")
            self.start_lstm_stream()
        else:
            self.lstm_stream_button.setText("Start LSTM Stream")
            self.stop_lstm_stream()

    def handle_lstm_prediction(self, result: dict):
        forecast = result.get('forecast')
        unc = result.get('uncertainty')
        delta = result.get('delta')
        pct = result.get('percent_change')
        last_close = result.get('last_close')
        ts = result.get('timestamp')
        self.lstm_forecast_label.setText(
            f"LSTM Stream: {forecast:.4f} (Δ={delta:+.4f}, {pct:+.2f}%) | Unc={unc:.4f} | Last={last_close:.4f} | t={ts}"
        )
        self.append_lstm_stream_output(result)

    def show_lstm_stream_output_window(self):
        if hasattr(self, '_lstm_stream_output_dialog') and self._lstm_stream_output_dialog is not None:
            self._lstm_stream_output_dialog.raise_()
            self._lstm_stream_output_dialog.activateWindow()
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QTabWidget, QWidget, QFormLayout, QSpinBox, QLabel, QHBoxLayout, QComboBox
        from PyQt6.QtCore import QTimer, Qt
        import psutil
        import os
        import pyqtgraph as pg
        dialog = QDialog(self)
        dialog.setWindowTitle("LSTM Stream Output")
        dialog.setGeometry(400, 400, 900, 600)
        main_layout = QVBoxLayout()

        tab_widget = QTabWidget()

        # --- Stream Output Tab ---
        output_tab = QWidget()
        output_layout = QVBoxLayout()
        # Dark mode toggle
        self._lstm_darkmode_checkbox = QPushButton("Enable Dark Mode")
        self._lstm_darkmode_checkbox.setCheckable(True)
        self._lstm_darkmode_checkbox.setChecked(False)
        output_layout.addWidget(self._lstm_darkmode_checkbox)
        table = QTableWidget(0, 7)
        table.setHorizontalHeaderLabels([
            "Timestamp", "Forecast", "Δ", "% Change", "Uncertainty", "Last Close", "Trend"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        output_layout.addWidget(table)
        # Resource usage label
        self._lstm_resource_label = QLabel("CPU: -- | RAM: -- | GPU: --")
        output_layout.addWidget(self._lstm_resource_label)
        output_tab.setLayout(output_layout)
        tab_widget.addTab(output_tab, "Stream Output")

        # --- Settings Tab (unchanged) ---
        settings_tab = QWidget()
        settings_layout = QFormLayout()

        # LSTM window size defaults by timeframe
        LSTM_WINDOW_DEFAULTS = {
            '5m': 60,
            '15m': 40,
            '30m': 30,
            '1h': 24,
            '4h': 20,
            '1d': 14,
            '1w': 8,
            '1M': 6,
        }

        interval_spin = QSpinBox()
        interval_spin.setRange(1, 3600)
        interval_spin.setValue(self.lstm_streamer.interval if self.lstm_streamer else 60)
        settings_layout.addRow(QLabel("Streaming Interval (seconds):"), interval_spin)
        window_spin = QSpinBox()
        window_spin.setRange(5, 200)
        window_spin.setValue(getattr(self.lstm_streamer, 'window_size', 30) if self.lstm_streamer else 30)
        settings_layout.addRow(QLabel("LSTM Window Size:"), window_spin)

        # Add a label to confirm current window size and timeframe
        window_confirm_label = QLabel()
        window_confirm_label.setStyleSheet("color: #007acc; font-weight: bold;")
        settings_layout.addRow(QLabel("Current LSTM Window Setting:"), window_confirm_label)

        def update_window_confirm_label():
            tf = self.timeframe_combo.currentText()
            window_val = window_spin.value()
            window_confirm_label.setText(f"Timeframe: <b>{tf}</b> | Window Size: <b>{window_val}</b>")

        # Dynamically update window size based on timeframe
        def update_lstm_window_spinbox():
            tf = self.timeframe_combo.currentText()
            default_window = LSTM_WINDOW_DEFAULTS.get(tf, 30)
            window_spin.setValue(default_window)
            update_window_confirm_label()
        self.timeframe_combo.currentTextChanged.connect(update_lstm_window_spinbox)
        window_spin.valueChanged.connect(update_window_confirm_label)
        # Initialize window size and label on UI setup
        update_lstm_window_spinbox()

        uncertainty_spin = QSpinBox()
        uncertainty_spin.setRange(1, 100)
        uncertainty_spin.setValue(getattr(self.lstm_streamer, 'uncertainty_iter', 20) if hasattr(self.lstm_streamer, 'uncertainty_iter') else 20)
        settings_layout.addRow(QLabel("Uncertainty Iterations:"), uncertainty_spin)
        feature_cols = []
        if hasattr(self, 'last_df_hist') and self.last_df_hist is not None:
            feature_cols = [c for c in self.last_df_hist.columns if c not in ['timestamp', 'close']]
        features_label = QLabel(", ".join(feature_cols) if feature_cols else "No features detected.")
        features_label.setWordWrap(True)
        features_label.setStyleSheet("background:#f0f0f0;padding:6px;border-radius:4px;")
        settings_layout.addRow(QLabel("Active Features/Indicators (all enabled):"), features_label)
        model_combo = None
        model_files = []
        if os.path.exists("Models"):
            model_files = [f for f in os.listdir("Models") if f.endswith(".pt")]
        if len(model_files) > 1:
            model_combo = QComboBox()
            model_combo.addItems(model_files)
            settings_layout.addRow(QLabel("LSTM Model File:"), model_combo)
        apply_btn = QPushButton("Apply")
        def apply_settings():
            new_interval = interval_spin.value()
            new_window = window_spin.value()
            new_uncertainty = uncertainty_spin.value()
            selected_model = None
            if model_combo is not None:
                selected_model = model_combo.currentText()
            self.update_lstm_stream_parameters(
                interval=new_interval,
                window_size=new_window,
                uncertainty_iter=new_uncertainty,
                features=feature_cols,
                model_file=selected_model
            )
        apply_btn.clicked.connect(apply_settings)
        settings_layout.addRow(apply_btn)

        # --- LSTM Hyperparameter Tuning Button ---
        tune_btn = QPushButton("Tune LSTM Hyperparameters")
        settings_layout.addRow(tune_btn)

        # --- Train LSTM from CSV Button ---
        train_csv_btn = QPushButton("Train LSTM from CSV")
        settings_layout.addRow(train_csv_btn)
        train_csv_btn.clicked.connect(self.train_lstm_from_csv)

        def show_lstm_tuning_dialog():
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QPushButton, QTextEdit, QProgressBar
            from PyQt6.QtCore import Qt, QThread, pyqtSignal
            class LSTMOptunaWorker(QThread):
                progress = pyqtSignal(int, int, float, dict)
                finished = pyqtSignal(object)
                error = pyqtSignal(str)
                def __init__(self, analyzer, df, n_trials):
                    super().__init__()
                    self.analyzer = analyzer
                    self.df = df
                    self.n_trials = n_trials
                def run(self):
                    try:
                        def cb(trial_num, n_trials, best_score, best_params):
                            self.progress.emit(trial_num, n_trials, best_score if best_score is not None else float('inf'), best_params if best_params is not None else {})
                        study = self.analyzer.optimize_lstm_with_optuna(self.df, n_trials=self.n_trials, progress_callback=cb)
                        self.finished.emit(study)
                    except Exception as e:
                        self.error.emit(str(e))
            # Dialog UI
            dialog = QDialog()
            dialog.setWindowTitle("LSTM Hyperparameter Tuning")
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Number of Optuna Trials:"))
            trial_spin = QSpinBox()
            trial_spin.setRange(5, 100)
            trial_spin.setValue(20)
            layout.addWidget(trial_spin)
            start_btn = QPushButton("Start Tuning")
            layout.addWidget(start_btn)
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            layout.addWidget(progress_bar)
            result_box = QTextEdit()
            result_box.setReadOnly(True)
            layout.addWidget(result_box)
            close_btn = QPushButton("Close")
            close_btn.setEnabled(False)
            layout.addWidget(close_btn)
            dialog.setLayout(layout)
            worker = None
            def on_progress(trial_num, n_trials, best_score, best_params):
                pct = int(100 * trial_num / n_trials)
                progress_bar.setValue(pct)
                txt = f"Trial {trial_num}/{n_trials}\nBest MAE: {best_score:.5f}\nBest Params: {json.dumps(best_params, indent=2)}"
                result_box.setPlainText(txt)
            def on_finished(study):
                # Only show results if at least one trial completed successfully
                completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
                if not completed_trials:
                    result_box.setPlainText("No successful trials were completed. Please check your data and try again.")
                else:
                    best = study.best_trial
                    txt = f"Tuning Complete!\nBest MAE: {best.value:.5f}\nBest Params:\n{json.dumps(best.params, indent=2)}"
                    result_box.setPlainText(txt)
                progress_bar.setValue(100)
                close_btn.setEnabled(True)
            def on_error(msg):
                result_box.setPlainText(f"Error: {msg}")
                close_btn.setEnabled(True)
            def start_tuning():
                nonlocal worker
                n_trials = trial_spin.value()
                result_box.setPlainText("Starting tuning...")
                progress_bar.setValue(0)
                close_btn.setEnabled(False)
                # Use last_df_hist for tuning
                df = getattr(self, 'last_df_hist', None)
                if df is None or df.empty:
                    result_box.setPlainText("No data available. Please fetch and analyze data first.")
                    close_btn.setEnabled(True)
                    return
                worker = LSTMOptunaWorker(self.ml_analyzer, df, n_trials)
                worker.progress.connect(on_progress)
                worker.finished.connect(on_finished)
                worker.error.connect(on_error)
                worker.start()
            start_btn.clicked.connect(start_tuning)
            close_btn.clicked.connect(dialog.accept)
            dialog.exec()
        tune_btn.clicked.connect(show_lstm_tuning_dialog)
        settings_tab.setLayout(settings_layout)
        tab_widget.addTab(settings_tab, "Settings")
        # Add tooltip to the Settings tab label
        settings_tooltip = (
            "<b>LSTM Stream Settings Controls:</b><br>"
            "<ul>"
            "<li><b>Streaming Interval (seconds):</b> How often to run the LSTM prediction.</li>"
            "<li><b>LSTM Window Size:</b> Number of time steps used for each LSTM prediction window.</li>"
            "<li><b>Uncertainty Iterations:</b> Number of MC dropout runs for uncertainty estimation.</li>"
            "<li><b>Active Features/Indicators:</b> Features used as input to the LSTM model.</li>"
            "<li><b>LSTM Model File:</b> (if multiple) Select which trained LSTM model to use.</li>"
            "<li><b>Apply:</b> Apply the above settings to the running LSTM stream.</li>"
            "</ul>"
        )
        tab_widget.setTabToolTip(1, settings_tooltip)

        main_layout.addWidget(tab_widget)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        main_layout.addWidget(close_btn)
        dialog.setLayout(main_layout)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        def on_close():
            # Save LSTM stream output chart data and interval settings
            import json
            save_data = {}
            # Save interval, window, uncertainty, and selected model if available
            try:
                interval = interval_spin.value()
                window = window_spin.value()
                uncertainty = uncertainty_spin.value()
                selected_model = model_combo.currentText() if model_combo is not None else None
                save_data['interval'] = interval
                save_data['window'] = window
                save_data['uncertainty'] = uncertainty
                save_data['model_file'] = selected_model
            except Exception:
                pass
            if save_data:
                with open('lstm_stream_settings.json', 'w') as f:
                    json.dump(save_data, f)
            self._lstm_stream_output_dialog = None
            self._lstm_stream_output_table = None
            self._lstm_resource_label = None
            if hasattr(self, '_lstm_resource_timer') and self._lstm_resource_timer is not None:
                self._lstm_resource_timer.stop()
                self._lstm_resource_timer = None
        dialog.finished.connect(on_close)

        # Restore settings if file exists
        import os, json
        if os.path.exists('lstm_stream_settings.json'):
            try:
                with open('lstm_stream_settings.json', 'r') as f:
                    settings = json.load(f)
                # Restore interval, window, uncertainty, model
                if 'interval' in settings:
                    interval_spin.setValue(settings['interval'])
                if 'window' in settings:
                    window_spin.setValue(settings['window'])
                if 'uncertainty' in settings:
                    uncertainty_spin.setValue(settings['uncertainty'])
                if 'model_file' in settings and model_combo is not None:
                    idx = model_combo.findText(settings['model_file'])
                    if idx >= 0:
                        model_combo.setCurrentIndex(idx)
            except Exception:
                pass

        self._lstm_stream_output_dialog = dialog
        self._lstm_stream_output_table = table
        # Sparkline data buffer
        self._lstm_sparkline_data = []
        self._lstm_resource_label = self._lstm_resource_label

        # --- Resource Usage Timer ---
        def update_resource_usage():
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu_str = "--"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    gpu_str = f"{gpu_mem:.2f}GB/{gpu_total:.2f}GB"
                else:
                    gpu_str = "N/A"
            except Exception:
                try:
                    import subprocess
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    if result.returncode == 0:
                        used, total = result.stdout.strip().split(',')
                        gpu_str = f"{float(used)/1024:.2f}GB/{float(total)/1024:.2f}GB"
                except Exception:
                    gpu_str = "--"
            self._lstm_resource_label.setText(f"CPU: {cpu:.1f}% | RAM: {ram:.1f}% | GPU: {gpu_str}")
        self._lstm_resource_timer = QTimer()
        self._lstm_resource_timer.timeout.connect(update_resource_usage)
        self._lstm_resource_timer.start(2000)

    def update_lstm_stream_parameters(self, interval=None, window_size=None, uncertainty_iter=None, features=None, model_file=None):
        from PyQt6.QtWidgets import QMessageBox
        if hasattr(self, 'lstm_streamer') and self.lstm_streamer is not None:
            self.lstm_streamer.stop()
            self.lstm_streamer.wait()
            def fetch_latest_ohlcv():
                df = getattr(self, 'last_df_hist', None)
                if df is not None and features is not None:
                    cols = ['timestamp'] + features + ['close']
                    df = df[[c for c in cols if c in df.columns]]
                return df
            lstm_model = self.ml_analyzer.model
            if model_file is not None:
                from src.analysis.lstm_regressor import LSTMRegressor
                import os
                model_path = os.path.join("Models", model_file)
                input_size = len(features) if features is not None else 0
                # --- PATCH: Check input size before loading ---
                try:
                    # Try to load state_dict to get expected input size
                    state_dict = None
                    import torch
                    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                    # Infer input_size from state_dict shape
                    weight_ih = state_dict.get('lstm.weight_ih_l0', None)
                    if weight_ih is not None:
                        expected_input_size = weight_ih.shape[1]
                        if input_size != expected_input_size:
                            QMessageBox.critical(self, "Model Load Error", f"Selected model expects {expected_input_size} features, but you selected {input_size}. Please select the correct features or retrain the model.")
                            return
                    lstm_model = LSTMRegressor.load(model_path, input_size=input_size)
                except Exception as e:
                    QMessageBox.critical(self, "Model Load Error", f"Failed to load LSTM model:\n{e}")
                    return  # Do not start streamer if model load fails
            self.lstm_streamer = LSTMDataStreamer(
                fetch_func=fetch_latest_ohlcv,
                lstm_model=lstm_model,
                window_size=window_size or getattr(self.lstm_streamer, 'window_size', 30),
                interval=interval or getattr(self.lstm_streamer, 'interval', 60)
            )
            if uncertainty_iter is not None:
                self.lstm_streamer.uncertainty_iter = uncertainty_iter
            self.lstm_streamer.prediction_ready.connect(self.handle_lstm_prediction)
            self.lstm_streamer.error_occurred.connect(lambda msg: self.lstm_forecast_label.setText(f"LSTM Stream Error: {msg}"))
            self.lstm_streamer.start()

    def append_lstm_stream_output(self, result: dict):
        if not hasattr(self, '_lstm_stream_output_table') or self._lstm_stream_output_table is None:
            return
        table = self._lstm_stream_output_table
        row = table.rowCount()
        table.insertRow(row)
        # --- Color coding and trend arrow ---
        ts = str(result.get('timestamp', ''))
        forecast = result.get('forecast', 0)
        delta = result.get('delta', 0)
        pct = result.get('percent_change', 0)
        unc = result.get('uncertainty', 0)
        # --- PATCH: Track previous close ---
        if not hasattr(self, '_lstm_prev_close'):
            self._lstm_prev_close = None
        prev_close = self._lstm_prev_close
        # Update for next call
        self._lstm_prev_close = result.get('last_close', 0)
        # Trend arrow
        if delta > 0.0001:
            trend = '↑'
            trend_color = Qt.GlobalColor.green
        elif delta < -0.0001:
            trend = '↓'
            trend_color = Qt.GlobalColor.red
        else:
            trend = '→'
            trend_color = Qt.GlobalColor.gray
        # Timestamp
        ts_item = QTableWidgetItem(ts)
        table.setItem(row, 0, ts_item)
        # Forecast
        forecast_item = QTableWidgetItem(f"{forecast:.4f}")
        table.setItem(row, 1, forecast_item)
        # Delta
        delta_item = QTableWidgetItem(f"{delta:+.4f}")
        delta_item.setForeground(Qt.GlobalColor.green if delta > 0 else Qt.GlobalColor.red if delta < 0 else Qt.GlobalColor.gray)
        table.setItem(row, 2, delta_item)
        # Percent change
        pct_item = QTableWidgetItem(f"{pct:+.2f}%")
        pct_item.setForeground(Qt.GlobalColor.green if pct > 0 else Qt.GlobalColor.red if pct < 0 else Qt.GlobalColor.gray)
        table.setItem(row, 3, pct_item)
        # Uncertainty
        unc_item = QTableWidgetItem(f"{unc:.4f}")
        if unc > 1.0:
            unc_item.setForeground(Qt.GlobalColor.red)
        elif unc > 0.5:
            unc_item.setForeground(Qt.GlobalColor.darkYellow)
        else:
            unc_item.setForeground(Qt.GlobalColor.darkGreen)
        table.setItem(row, 4, unc_item)
        # Last close (use previous close)
        last_item = QTableWidgetItem(f"{prev_close:.4f}" if prev_close is not None else "--")
        table.setItem(row, 5, last_item)
        # Trend
        trend_item = QTableWidgetItem(trend)
        trend_item.setForeground(trend_color)
        table.setItem(row, 6, trend_item)
        table.scrollToBottom()
        # --- Sparkline update ---
        if hasattr(self, '_lstm_sparkline_data'):
            self._lstm_sparkline_data.append((ts, forecast, prev_close if prev_close is not None else 0, unc))
            # Keep only last 100 points
            if len(self._lstm_sparkline_data) > 100:
                self._lstm_sparkline_data = self._lstm_sparkline_data[-100:]
            x = list(range(len(self._lstm_sparkline_data)))
            y_forecast = [d[1] for d in self._lstm_sparkline_data]
            y_actual = [d[2] for d in self._lstm_sparkline_data]
            y_unc = [d[1] + d[3] for d in self._lstm_sparkline_data]
            if hasattr(self, '_lstm_sparkline_forecast') and self._lstm_sparkline_forecast is not None:
                self._lstm_sparkline_forecast.setData(x, y_forecast)
            if hasattr(self, '_lstm_sparkline_actual') and self._lstm_sparkline_actual is not None:
                self._lstm_sparkline_actual.setData(x, y_actual)
            if hasattr(self, '_lstm_sparkline_unc') and self._lstm_sparkline_unc is not None:
                self._lstm_sparkline_unc.setData(x, y_unc)

    def closeEvent(self, event):
        if hasattr(self, 'lstm_streamer') and self.lstm_streamer is not None:
            self.stop_lstm_stream()
        event.accept()

    def show_screener_window(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QLineEdit, QLabel, QHBoxLayout, QMessageBox
        from PyQt6.QtCore import Qt
        import threading

        dialog = QDialog(self)
        dialog.setWindowTitle("Token/Pairs Screener")
        dialog.setGeometry(300, 300, 1200, 700)
        layout = QVBoxLayout()

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_input = QLineEdit()
        filter_input.setPlaceholderText("Filter by pair, exchange, or signal...")
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(filter_input)
        layout.addLayout(filter_layout)

        # Table
        columns = ["Pair", "Exchange", "Price", "Change %", "Volume", "ML Signal"]
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)

        # Status label
        status_label = QLabel("Loading...")
        layout.addWidget(status_label)

        # Print number of pairs for debug
        print(f"[Screener] Number of pairs in pair_sources: {len(self.pair_sources)}")
        if not self.pair_sources:
            QMessageBox.warning(self, "Screener", "No pairs available to screen. Please fetch data first.")
            return

        # Fetch and populate data in a background thread
        def fetch_and_populate():
            rows = []
            pairs = list(self.pair_sources.items())[:50]  # Limit for speed
            for idx, (pair, sources) in enumerate(pairs):
                exchange = sources[0]
                try:
                    ohlcv = fetch_ohlcv(exchange, pair, "1h", limit=30)
                    if ohlcv:
                        last = ohlcv[-1]
                        price = last[4]
                        open_ = last[1]
                        change_pct = ((price - open_) / open_ * 100) if open_ else 0
                        volume = last[5]
                        # Prepare DataFrame for ML
                        import pandas as pd
                        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        # Add indicators (use your helper)
                        try:
                            df = self.add_technical_indicators(df)
                        except Exception as e:
                            print(f"[Screener] Error adding indicators for {pair}: {e}")
                        # Run ML signal (fast, no LLM, no heavy features)
                        try:
                            ml_result = self.ml_analyzer.analyze_market_data(
                                symbol=pair,
                                klines_data=df,
                                additional_context={"timeframe": "1h"}
                            )
                            ml_signal = ml_result.get("recommendation", "--")
                        except Exception as e:
                            print(f"[Screener] Error running ML for {pair}: {e}")
                            ml_signal = "--"
                        rows.append([pair, exchange, f"{price:.4f}", f"{change_pct:+.2f}%", f"{volume:.0f}", ml_signal, change_pct])
                    else:
                        print(f"[Screener] No OHLCV data for {pair} on {exchange}")
                        rows.append([pair, exchange, "--", "--", "--", "--", float('-inf')])
                except Exception as e:
                    print(f"[Screener] Error fetching {pair} on {exchange}: {e}")
                    rows.append([pair, exchange, "--", "--", "--", "--", float('-inf')])
                # Update status
                self.run_on_main_thread(lambda idx=idx: status_label.setText(f"Loaded {idx+1}/{len(pairs)} pairs..."))
            # Sort: ML Signal (BUY > SELL > others), then by change % descending
            def ml_sort_key(row):
                signal = row[5].upper()
                if "BUY" in signal:
                    return (0, -row[6])
                elif "SELL" in signal:
                    return (1, -row[6])
                else:
                    return (2, -row[6])
            rows.sort(key=ml_sort_key)
            # Populate table in the main thread
            def update_table():
                table.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    for j, val in enumerate(row[:6]):
                        item = QTableWidgetItem(str(val))
                        if j == 3:  # Change %
                            item.setForeground(Qt.GlobalColor.green if "+" in str(val) else Qt.GlobalColor.red)
                        if j == 5:  # ML Signal
                            if "BUY" in str(val).upper():
                                item.setForeground(Qt.GlobalColor.green)
                            elif "SELL" in str(val).upper():
                                item.setForeground(Qt.GlobalColor.red)
                            else:
                                item.setForeground(Qt.GlobalColor.gray)
                        table.setItem(i, j, item)
                status_label.setText("Done.")
            self.run_on_main_thread(update_table)
        threading.Thread(target=fetch_and_populate, daemon=True).start()

        # Filter logic
        def filter_table():
            text = filter_input.text().lower()
            for row in range(table.rowCount()):
                show = any(text in (table.item(row, col).text().lower() if table.item(row, col) else "") for col in range(table.columnCount()))
                table.setRowHidden(row, not show)
        filter_input.textChanged.connect(filter_table)

        # Double-click to select pair
        def on_row_double_clicked(row, col):
            pair = table.item(row, 0).text()
            self.select_pair_button.setText(pair)
            dialog.accept()
            self.fetch_data()
        table.cellDoubleClicked.connect(on_row_double_clicked)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec()

    def run_on_main_thread(self, func):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, func)

    def train_lstm_from_csv(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QSpinBox, QPushButton, QHBoxLayout
        import pandas as pd
        import os
        import numpy as np
        # --- Dialog for config ---
        dialog = QDialog(self)
        dialog.setWindowTitle("Train LSTM from CSV(s)")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select one or more CSV files with market data."))
        # Window size
        window_label = QLabel("LSTM Window Size (optimal suggested):")
        window_spin = QSpinBox()
        window_spin.setRange(5, 200)
        window_spin.setValue(30)  # Optimal default
        layout.addWidget(window_label)
        layout.addWidget(window_spin)
        # Epochs
        epochs_label = QLabel("Epochs (optimal suggested):")
        epochs_spin = QSpinBox()
        epochs_spin.setRange(5, 200)
        epochs_spin.setValue(30)  # Optimal default
        layout.addWidget(epochs_label)
        layout.addWidget(epochs_spin)
        # File select
        file_btn = QPushButton("Select CSV Files")
        file_label = QLabel("No files selected.")
        layout.addWidget(file_btn)
        layout.addWidget(file_label)
        selected_files = []
        def select_files():
            files, _ = QFileDialog.getOpenFileNames(dialog, "Select Market Data CSVs", "", "CSV Files (*.csv)")
            if files:
                selected_files.clear()
                selected_files.extend(files)
                file_label.setText("\n".join(os.path.basename(f) for f in files))
            else:
                file_label.setText("No files selected.")
        file_btn.clicked.connect(select_files)
        # Train button
        train_btn = QPushButton("Train LSTM")
        layout.addWidget(train_btn)
        # Status label
        status_label = QLabel()
        layout.addWidget(status_label)
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        # --- Training logic ---
        def train():
            if not selected_files:
                QMessageBox.critical(dialog, "No Files", "Please select at least one CSV file.")
                return
            window = window_spin.value()
            epochs = epochs_spin.value()
            all_features = []
            all_targets = []
            feature_cols = None
            for file_path in selected_files:
                try:
                    df = pd.read_csv(file_path)
                    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
                    if not required_cols.issubset(df.columns):
                        status_label.setText(f"File {os.path.basename(file_path)} missing required columns.")
                        return
                    df = self.add_technical_indicators(df)
                    # Use same features for all files
                    if feature_cols is None:
                        feature_cols = [c for c in df.columns if c not in ['timestamp', 'close']]
                    features = df[feature_cols].values.astype(np.float32)
                    targets = df['close'].values.astype(np.float32)
                    from src.analysis.lstm_regressor import LSTMRegressor
                    X, y = LSTMRegressor.create_windowed_dataset(features, targets, window)
                    if X.shape[0] == 0:
                        status_label.setText(f"File {os.path.basename(file_path)}: Not enough data for window size {window}.")
                        return
                    all_features.append(X)
                    all_targets.append(y)
                except Exception as e:
                    status_label.setText(f"Error processing {os.path.basename(file_path)}: {e}")
                    return
            # Concatenate all
            X = np.concatenate(all_features, axis=0)
            y = np.concatenate(all_targets, axis=0)
            input_size = X.shape[2] if X.ndim == 3 else len(feature_cols)
            try:
                from src.analysis.lstm_regressor import LSTMRegressor
                model = LSTMRegressor(input_size=input_size)
                status_label.setText("Training LSTM...")
                dialog.repaint()
                model.fit(X, y, epochs=epochs, batch_size=32, lr=1e-3, device=None, verbose=True)
                model_path = os.path.join("Models", "lstm_from_csv.pt")
                model.save(model_path)
                QMessageBox.information(dialog, "LSTM Training", f"LSTM trained and saved to {model_path}")
                self.ml_analyzer.model = model  # Use for forecasting
                status_label.setText(f"Training complete. Model saved to {model_path} and ready for forecasting.")
            except Exception as e:
                QMessageBox.critical(dialog, "Training Error", f"Error during LSTM training: {e}")
                status_label.setText(f"Error during training: {e}")
        train_btn.clicked.connect(train)
        dialog.exec()

    def download_all_pairs_all_timeframes(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QProgressDialog, QMessageBox
        import time
        import traceback
        # Confirm dialog
        confirm = QMessageBox.question(self, "Download All Pairs", "This will download OHLCV data for ALL pairs from all connected exchanges and all timeframes. This may take a long time and use significant bandwidth. Continue?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        # Gather all pairs and exchanges
        okx_pairs = set()
        try:
            from Data.okx_public_data import fetch_okx_ticker
            tickers = fetch_okx_ticker(None, all_markets=True)
            okx_pairs = set(
                t['instId'].replace("-", "/")
                for t in tickers if 'instId' in t and t['instId'].endswith('-USDT')
            )
        except Exception as e:
            okx_pairs = set()
        ccxt_pairs_by_exchange = {}
        for exch in self.ccxt_exchanges:
            try:
                pairs = self.get_ccxt_pairs(exch)
                ccxt_pairs_by_exchange[exch] = pairs
            except Exception as e:
                ccxt_pairs_by_exchange[exch] = set()
        # Timeframes
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        # Count total tasks
        total_tasks = len(okx_pairs) * len(timeframes) + sum(len(pairs) for pairs in ccxt_pairs_by_exchange.values()) * len(timeframes)
        progress = QProgressDialog("Downloading all pairs...", "Cancel", 0, total_tasks, self)
        progress.setWindowTitle("Download Progress")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        # Download OKX pairs
        from src.utils.data_storage import save_ohlcv
        from src.utils.helpers import fetch_ohlcv
        task = 0
        errors = []
        for pair in okx_pairs:
            for tf in timeframes:
                if progress.wasCanceled():
                    progress.close()
                    QMessageBox.information(self, "Download Cancelled", "Download was cancelled by the user.")
                    return
                try:
                    ohlcv = fetch_ohlcv("OKX", pair, tf, limit=1000)
                    if ohlcv:
                        save_ohlcv(pair, tf, ohlcv)
                    else:
                        errors.append(f"No data for {pair} ({tf}) [OKX]")
                except Exception as e:
                    errors.append(f"Error {pair} ({tf}) [OKX]: {e}")
                task += 1
                progress.setValue(task)
                QApplication.processEvents()
        # Download CCXT pairs
        for exch, pairs in ccxt_pairs_by_exchange.items():
            for pair in pairs:
                for tf in timeframes:
                    # Map to valid CCXT timeframe for this exchange
                    mapped_tf = self.get_ccxt_timeframe(exch, tf)
                    # Check if mapped_tf is valid for this exchange
                    try:
                        import ccxt
                        exchange_class = getattr(ccxt, exch.lower())
                        exchange = exchange_class()
                        valid_timeframes = set(getattr(exchange, 'timeframes', {}).keys())
                    except Exception:
                        valid_timeframes = set()
                    if mapped_tf not in valid_timeframes:
                        errors.append(f"Skipped {pair} ({tf}) [CCXT:{exch}]: Invalid timeframe '{tf}'. Must be one of {valid_timeframes}")
                        continue
                    if progress.wasCanceled():
                        progress.close()
                        QMessageBox.information(self, "Download Cancelled", "Download was cancelled by the user.")
                        return
                    try:
                        ohlcv = fetch_ohlcv(f"ccxt:{exch}", pair, mapped_tf, limit=1000)
                        if ohlcv:
                            save_ohlcv(pair, mapped_tf, ohlcv)
                        else:
                            errors.append(f"No data for {pair} ({mapped_tf}) [CCXT:{exch}]")
                    except Exception as e:
                        errors.append(f"Error {pair} ({mapped_tf}) [CCXT:{exch}]: {e}")
                    task += 1
                    progress.setValue(task)
                    QApplication.processEvents()
        progress.close()
        if errors:
            QMessageBox.warning(self, "Download Complete (with errors)", f"Some pairs failed to download.\n\n" + "\n".join(errors[:20]) + ("\n..." if len(errors) > 20 else ""))
        else:
            QMessageBox.information(self, "Download Complete", "All pairs downloaded successfully.")

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD, MACD Signal, and MACD Histogram using pandas.
    Returns a DataFrame with columns: 'macd', 'macd_signal', 'macd_hist'.
    """
    if close.isnull().all() or close.dropna().shape[0] < slow:
        return pd.DataFrame({
            'macd': [np.nan] * len(close),
            'macd_signal': [np.nan] * len(close),
            'macd_hist': [np.nan] * len(close)
        }, index=close.index)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist
    }, index=close.index)

# --- Ollama health check ---
def ollama_is_running() -> bool:
    """Check if Ollama is running locally by pinging the /api/tags endpoint."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def call_ollama_llm(prompt: str, model: str = "llama2", temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Call the local Ollama LLM API with a prompt and return the response.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[No response from Ollama]")
    except Exception as e:
        return f"[Ollama API error: {e}]"

from PyQt6.QtCore import QThread, pyqtSignal
class TickerFetcher(QThread):
    data_fetched = pyqtSignal(dict)
    def __init__(self, get_symbol_func):
        super().__init__()
        self.get_symbol_func = get_symbol_func
        self._running = True
    def run(self):
        import time
        from Data.okx_public_data import fetch_okx_ticker
        while self._running:
            symbol = self.get_symbol_func()
            try:
                ticker = fetch_okx_ticker(symbol)
                if ticker:
                    self.data_fetched.emit(ticker)
                else:
                    self.data_fetched.emit({})
            except Exception:
                self.data_fetched.emit({})
            time.sleep(1)
    def stop(self):
        self._running = False

class LSTMTrainingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    def __init__(self, X, y, input_size, model_path, epochs=30, batch_size=32, lr=1e-3):
        super().__init__()
        from src.analysis.lstm_regressor import LSTMRegressor
        self.X = X
        self.y = y
        self.input_size = input_size
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.LSTMRegressor = LSTMRegressor
    def run(self):
        log_html = "<b>[ML Training]</b> Starting LSTM training...<br>"
        model = self.LSTMRegressor(input_size=self.input_size)
        def progress_callback(epoch, epochs, loss):
            nonlocal log_html
            log_html += f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}<br>"
            self.progress.emit(log_html)
        model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, device=None, verbose=False, progress_callback=progress_callback)
        model.save(self.model_path)
        log_html += f"<b>[ML Training]</b> LSTM model saved to {self.model_path}<br>"
        self.progress.emit(log_html)
        self.finished.emit(model)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 