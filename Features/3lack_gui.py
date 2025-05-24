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
    QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QAbstractItemView, QDoubleSpinBox, QSpinBox, QInputDialog, QMessageBox, QProgressBar, QCompleter, QDialog, QListWidget, QDialogButtonBox, QCheckBox, QTabWidget, QScrollArea, QToolButton, QProgressDialog
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
        self.algorithm_combo.addItems(["RandomForest", "GradientBoosting", "SVM", "XGBoost", "Ensemble (Voting)", "Ensemble (Stacking)"])
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

        # LLM Output
        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)
        # Enable full text selection and copying for easy copy-paste
        self.llm_output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )

        # Test API Button (was OKX API Test Button)
        self.test_api_button = QPushButton("Test API")
        self.test_api_button.clicked.connect(self.run_api_test)

        # Download All Timeframes Button
        self.download_all_button = QPushButton("Download All Timeframes")
        self.download_all_button.clicked.connect(self.download_all_timeframes_for_selected_pair)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

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
        main_layout.addWidget(self.download_all_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.backtest_button)
        main_layout.addWidget(self.show_chart_button)
        main_layout.addWidget(self.test_api_button)
        main_layout.addWidget(self.current_label)
        main_layout.addWidget(self.reversal_label)  # Add reversal label to layout
        main_layout.addWidget(QLabel("Analysis & Recommendations:"))
        main_layout.addWidget(self.llm_output)

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
            self.llm_output.setText("Please select a pair.")
            self.reversal_label.setText("Reversal: --")
            self.reversal_label.setStyleSheet("color: black; font-weight: bold; font-size: 16px;")
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
            self.llm_output.setText("")
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
            self.llm_output.setText("Failed to fetch OHLCV data. Please check the symbol, timeframe, or your internet connection.")
            self.reversal_label.setText("Reversal: --")
            self.reversal_label.setStyleSheet("color: black; font-weight: bold; font-size: 16px;")

    def run_analysis(self):
        """Run analysis using MLAnalyzer only."""
        import os
        import joblib
        mode = self.analysis_mode_combo.currentText()
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}"
        else:
            self.llm_output.setHtml("<b>[Analysis]</b> Please select an exchange and pair.")
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
                self.llm_output.setHtml("<b>[Analysis]</b> Failed to load hyperparameter-tuned model. Using default.")
        elif os.path.exists(model_path):
            try:
                pipeline = joblib.load(model_path)
                self.ml_analyzer.pipeline_ = pipeline
            except Exception:
                self.llm_output.setHtml("<b>[Analysis]</b> Failed to load regular model. Using default.")
        # Continue with analysis as before
        from src.utils.data_storage import load_ohlcv
        df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
            self.llm_output.setHtml("<b>[Analysis]</b> No historical data available for this symbol/timeframe.")
            return
        try:
            df_hist = self.add_technical_indicators(df_hist)
        except ValueError as e:
            self.llm_output.setHtml(f"<b>[Analysis]</b> {e}")
            return
        self.ml_analyzer.set_custom_model_class(None)
        output = ""
        ml_result = None
        if self.ml_analyzer.algorithm != self.selected_algorithm:
            self.on_algorithm_changed(self.selected_algorithm)
        if mode in ["Machine Learning", "Both"]:
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
        self.llm_output.setHtml(output)
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

Do NOT provide trading recommendations, entry/exit prices, or any rationale for action. Do NOT include verbose explanations or general advice. Only summarize the technical state of the market in a factual, data-driven manner.

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

    def run_api_test(self):
        """Test OKX and Blofin API endpoints and display results in the LLM output box."""
        from Data.okx_public_data import test_api
        import pprint
        self.update_exchange_status()
        result = test_api()
        html = "<b>API Connectivity Test Results</b><br>"
        for ex, ex_result in result.items():
            html += f"<b>{ex.upper()}:</b> Status: {ex_result['status']}<br>"
            for k, v in ex_result['details'].items():
                html += f"&nbsp;&nbsp;<b>{k}:</b> {pprint.pformat(v)}<br>"
        self.llm_output.setHtml(html)

    def ohlcv_to_df(self, ohlcv_data):
        # ohlcv_data: List[List[str]] (from table)
        df = pd.DataFrame(ohlcv_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        # Convert numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Convert timestamp to numeric if needed
        return df

    def format_ml_analysis_html(self, result: dict) -> str:
        """
        Format the ML analysis result as a comprehensive, visually clear HTML summary for display, using a structured HTML list. Exclude Direction and Price Action.
        """
        if not isinstance(result, dict):
            return f"<b>[ML Analysis]</b> Invalid result."
        if 'error' in result:
            return f"<b>[ML Analysis Error]</b> {result['error']}"
        def color_for_recommendation(rec):
            return 'green' if rec == 'BUY' else 'red' if rec == 'SELL' else 'gray'
        def color_for_sentiment(sent):
            return 'green' if sent == 'BULLISH' else 'red' if sent == 'BEARISH' else 'gray'
        def fmt_price(val):
            try:
                return f"{float(val):,.4f}"
            except Exception:
                return str(val)
        html = "<ul style='font-family:Arial,sans-serif;font-size:13px;'>"
        html += f"<li><b>Recommendation:</b> <span style='color:{color_for_recommendation(result.get('recommendation'))};font-weight:bold'>{result.get('recommendation','')}</span></li>"
        html += f"<li><b>Confidence:</b> <span style='color:gold;font-weight:bold'>{result.get('confidence',0):.2f}%</span></li>"
        html += f"<li><b>Current Price:</b> <span style='color:gold;font-weight:bold'>{fmt_price(result.get('current_price','N/A'))}</span></li>"
        html += f"<li><b>Sentiment:</b> <span style='color:{color_for_sentiment(result.get('metrics',{}).get('market_sentiment',''))};'>{result.get('metrics',{}).get('market_sentiment','N/A')}</span></li>"
        html += f"<li><b>Timeframe Suitability:</b> {result.get('metrics',{}).get('timeframe_suitability','N/A')}</li>"
        # Support/Resistance
        support_levels = ', '.join(fmt_price(s) for s in result.get('support_levels', []))
        resistance_levels = ', '.join(fmt_price(r) for r in result.get('resistance_levels', []))
        html += f"<li><b>Support Levels:</b> {support_levels}</li>"
        html += f"<li><b>Resistance Levels:</b> {resistance_levels}</li>"
        # Limit Orders (condensed)
        lo = result.get('limit_orders', {})
        if lo:
            orders = []
            for order_type in ['entry', 'take_profit', 'stop_loss']:
                for o in lo.get(order_type, []):
                    confidence = str(o.get('confidence', '')).upper()
                    if confidence == 'LOW':
                        continue
                    orders.append(f"{order_type.title()}: {o.get('type','')} @ {fmt_price(o.get('price',''))} ({confidence})")
            if orders:
                html += "<li><b>Orders:</b><ul>"
                for order in orders:
                    html += f"<li>{order}</li>"
                html += "</ul></li>"
        html += "</ul>"
        return html

    def add_technical_indicators(self, df: pd.DataFrame, min_rows: int = 35) -> pd.DataFrame:
        """
        Add a suite of technical indicators to the OHLCV DataFrame using pandas_ta default column names for all indicators.
        """
        import pandas_ta as ta
        import numpy as np
        import pandas as pd
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame is empty or missing required OHLCV columns.")
        if df['close'].isnull().all():
            raise ValueError("No valid 'close' price data available for indicator calculation.")
        df = df.rename(columns={col: col.lower() for col in df.columns})
        # --- Standard indicators ---
        indicators = [
            ta.ema(df['close'], length=9),
            ta.ema(df['close'], length=20),
            ta.ema(df['close'], length=50),
            ta.ema(df['close'], length=100),
            ta.macd(df['close']),
            ta.adx(df['high'], df['low'], df['close']),
            ta.rsi(df['close'], length=14),
            ta.stoch(df['high'], df['low'], df['close']),
            ta.willr(df['high'], df['low'], df['close'], length=14),
            ta.cci(df['high'], df['low'], df['close'], length=20),
            ta.roc(df['close'], length=12),
            ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14),
            ta.bbands(df['close'], length=20),
            ta.atr(df['high'], df['low'], df['close'], length=14),
            ta.obv(df['close'], df['volume']),
            ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20),
        ]
        for ind in indicators:
            if ind is not None:
                df = pd.concat([df, ind], axis=1)
        # --- Advanced features ---
        # Candle pattern flags
        candle_patterns = ['cdl_engulfing', 'cdl_hammer', 'cdl_doji', 'cdl_morningstar', 'cdl_eveningstar']
        for pat in candle_patterns:
            try:
                df[pat] = getattr(ta, pat)(df['open'], df['high'], df['low'], df['close'])
            except Exception:
                df[pat] = 0
        # Rolling returns and volatility
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        # Price/MA ratios
        df['close_ema9_ratio'] = df['close'] / df['close'].ewm(span=9, adjust=False).mean()
        df['close_ema20_ratio'] = df['close'] / df['close'].ewm(span=20, adjust=False).mean()
        # Time-of-day features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
        else:
            df['hour'] = 0
            df['dayofweek'] = 0
        # --- More advanced features ---
        # SuperTrend
        try:
            st = ta.supertrend(df['high'], df['low'], df['close'])
            if isinstance(st, pd.DataFrame):
                df['supertrend'] = st.iloc[:, 0]
        except Exception:
            df['supertrend'] = 0
        # Keltner Channel width
        try:
            kc = ta.kc(df['high'], df['low'], df['close'])
            if isinstance(kc, pd.DataFrame):
                df['kc_width'] = kc['KCU_20_2.0'] - kc['KCL_20_2.0']
        except Exception:
            df['kc_width'] = 0
        # Donchian Channel breakout
        try:
            dc = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
            if isinstance(dc, pd.DataFrame):
                df['donchian_breakout'] = (df['close'] >= dc['DONCHIANU_20_20']).astype(int) - (df['close'] <= dc['DONCHIANL_20_20']).astype(int)
        except Exception:
            df['donchian_breakout'] = 0
        # TRIX
        try:
            df['trix'] = ta.trix(df['close'])
        except Exception:
            df['trix'] = 0
        # Ultimate Oscillator
        try:
            df['uo'] = ta.uo(df['high'], df['low'], df['close'])
        except Exception:
            df['uo'] = 0
        # Z-Score of price
        df['zscore_close'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        # Pivot Points (Classic)
        try:
            pivots = ta.pivot('classic', df['high'], df['low'], df['close'])
            if isinstance(pivots, pd.DataFrame):
                df['pivot'] = pivots['P']
        except Exception:
            df['pivot'] = 0
        # Fractals
        try:
            fr = ta.fractals(df['high'], df['low'])
            if isinstance(fr, pd.DataFrame):
                df['fractal_high'] = fr['FRACTALS_HI_2']
                df['fractal_low'] = fr['FRACTALS_LO_2']
        except Exception:
            df['fractal_high'] = 0
            df['fractal_low'] = 0
        # Distance to recent high/low
        df['dist_to_high'] = df['close'] / df['high'].rolling(20).max() - 1
        df['dist_to_low'] = df['close'] / df['low'].rolling(20).min() - 1
        # Consecutive up/down closes
        df['consec_up'] = (df['close'] > df['close'].shift(1)).astype(int).groupby((df['close'] <= df['close'].shift(1)).cumsum()).cumsum()
        df['consec_down'] = (df['close'] < df['close'].shift(1)).astype(int).groupby((df['close'] >= df['close'].shift(1)).cumsum()).cumsum()
        # Volume spike flag
        df['vol_spike'] = (df['volume'] > 2 * df['volume'].rolling(20).mean()).astype(int)
        # Rolling Sharpe ratio
        df['rolling_sharpe'] = df['return_1'].rolling(20).mean() / (df['return_1'].rolling(20).std() + 1e-9)
        # Is weekend
        if isinstance(df.index, pd.DatetimeIndex):
            df['is_weekend'] = df.index.dayofweek >= 5
        else:
            df['is_weekend'] = 0
        df = df.dropna().reset_index(drop=True)
        valid_rows = len(df.dropna(subset=["open", "high", "low", "close", "volume"]))
        if valid_rows < min_rows:
            raise ValueError(f"Not enough valid data for indicator calculation. At least {min_rows} rows are required after cleaning. Valid rows: {valid_rows}")
        return df

    def train_model(self):
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import QThread, pyqtSignal, QObject
        import os
        import joblib
        # Use a default or dynamically determined rows value
        rows = 1000  # Default value, or compute based on timeframe/algorithm if needed
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}/USDT"
        else:
            self.llm_output.append("<b>[ML Training]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
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
            self.llm_output.setHtml("<b>[ML Training]</b> No historical data available for this symbol/timeframe.")
            return
        valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
        if valid_rows < indicator_lookback:
            self.llm_output.setHtml(
                f"<b>[ML Training]</b> Not enough valid data for indicator calculation. At least {indicator_lookback} rows are required after cleaning. Valid rows: {valid_rows}"
            )
            return
        usable_rows = min(rows, valid_rows - indicator_lookback)
        if usable_rows < 1:
            self.llm_output.setHtml(
                f"<b>[ML Training]</b> Not enough data after indicator lookback. Try lowering the rows setting or using a higher timeframe."
            )
            return
        if usable_rows < rows:
            self.llm_output.append(
                f"<b>[ML Training]</b> Only {usable_rows} rows available for training after indicator calculation. Proceeding with available data."
            )
        try:
            df_hist = self.add_technical_indicators(df_hist, min_rows=usable_rows)
        except ValueError as e:
            self.llm_output.setHtml(f"<b>[ML Training]</b> {e}")
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
                            self.llm_output.append(f"<b>[ML Training]</b> Hyperparameter-tuned model saved to {model_path_optuna}")
                        else:
                            self.llm_output.append("<b>[ML Training]</b> Warning: No pipeline_ attribute found after Optuna tuning.")
                    except Exception as e:
                        self.llm_output.append(f"<b>[ML Training]</b> Error saving Optuna-tuned model: {e}")
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
                    self.llm_output.setHtml(html)
                def on_error(msg):
                    progress.close()
                    self.llm_output.setHtml(f"<b>[ML Training]</b> Error during tuning: {msg}")
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
                self.llm_output.append(f"<b>[ML Training]</b> Regular model saved to {model_path}")
            else:
                self.llm_output.append("<b>[ML Training]</b> Warning: No pipeline_ attribute found after training.")
        except Exception as e:
            self.llm_output.append(f"<b>[ML Training]</b> Error saving regular model: {e}")
        self.llm_output.setHtml(self.format_ml_training_result_html(results))

    def format_ml_training_result_html(self, result: dict) -> str:
        """
        Format the ML training result as a human-friendly HTML summary for display, with color-coded metrics.
        Args:
            result (dict): The result dictionary from model training.
        Returns:
            str: HTML-formatted summary.
        """
        if not isinstance(result, dict):
            return "<b>[ML Training]</b> Invalid result."
        if 'error' in result:
            return f"<b>[ML Training Error]</b> {result['error']}"

        def color_metric(val: float) -> str:
            if val >= 0.8:
                return 'green'
            elif val >= 0.6:
                return 'gold'
            else:
                return 'red'

        html = "<ul style='font-family:Arial,sans-serif;font-size:13px;'>"
        for k, v in result.items():
            if isinstance(v, float):
                html += f"<li><b>{k.replace('_', ' ').capitalize()}:</b> <span style='color:{color_metric(v)};font-weight:bold'>{v:.4f}</span></li>"
            elif isinstance(v, int):
                html += f"<li><b>{k.replace('_', ' ').capitalize()}:</b> <span style='color:gold;font-weight:bold'>{v:,}</span></li>"
            elif isinstance(v, dict):
                html += f"<li><b>{k.replace('_', ' ').capitalize()}:</b><ul>"
                for k2, v2 in v.items():
                    html += f"<li>{k2}: {v2}</li>"
                html += "</ul></li>"
            elif isinstance(v, list) and v and all(isinstance(x, (float, int)) for x in v):
                formatted = ', '.join(f"{x:.4f}" for x in v)
                html += f"<li><b>{k.replace('_', ' ').capitalize()}:</b> <span style='font-family:monospace'>{formatted}</span></li>"
            else:
                html += f"<li><b>{k.replace('_', ' ').capitalize()}:</b> {v}</li>"
        html += "</ul>"
        return html

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
        # Add to main layout (assume it's the parent of data_table)
        self.centralWidget().layout().addWidget(self.pair_menu_widget)
        # Fetch OKX pairs
        okx_pairs = set()
        try:
            tickers = fetch_okx_ticker(None, all_markets=True)
            okx_pairs = set(
                t['instId'].replace("-", "/")
                for t in tickers if 'instId' in t and t['instId'].endswith('-USDT')
            )
        except Exception as e:
            okx_pairs = set([f"Error loading OKX pairs: {e}"])
        # Merge and mark source (do not fetch CCXT pairs here)
        all_pairs_set = okx_pairs
        self.pair_sources = {}
        for p in all_pairs_set:
            srcs = []
            if p in okx_pairs:
                srcs.append("OKX")
            self.pair_sources[p] = srcs
        # Display as: BTC/USDT [OKX]
        self.all_pairs = [f"{p} [{','.join(self.pair_sources[p])}]" for p in sorted(all_pairs_set)]
        self.pair_list.addItems(self.all_pairs)
        self.pair_list.itemClicked.connect(self.select_pair_from_menu)
        self.pair_search_input.textChanged.connect(self.filter_pair_list)
        self.exchange_combo.currentTextChanged.connect(lambda: self.filter_pair_list(self.pair_search_input.text()))

    def filter_pair_list(self, text: str) -> None:
        if not hasattr(self, 'pair_list') or not hasattr(self, 'all_pairs'):
            return
        search_text = text.strip().lower()
        selected_exchange = self.exchange_combo.currentText() if hasattr(self, 'exchange_combo') else "All"
        self.pair_list.clear()
        # If a CCXT exchange is selected, dynamically load its pairs
        if selected_exchange.startswith("CCXT:"):
            ccxt_name = selected_exchange.split(":", 1)[1]
            ccxt_pairs = self.get_ccxt_pairs(ccxt_name)
            for pair in sorted(ccxt_pairs):
                if search_text and search_text not in pair.lower():
                    continue
                self.pair_list.addItem(f"{pair} [CCXT:{ccxt_name}]")
            return
        # Otherwise, filter the preloaded pairs
        for label in self.all_pairs:
            pair = label.split('[')[0].strip().lower()
            srcs = label.split('[')[-1].strip('] ').split(',') if '[' in label else []
            if search_text and search_text not in pair:
                continue
            if selected_exchange != "All" and selected_exchange not in srcs:
                continue
            self.pair_list.addItem(label)

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

    def download_all_timeframes_for_selected_pair(self):
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
        if not pair or pair == "SELECT PAIR":
            self.llm_output.append("<b>[Download]</b> Please select a pair first.")
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
            self.llm_output.append(f"<b>[Download]</b> Exchange {exchange} not supported for bulk download.")
            self.progress_bar.setVisible(False)
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
                self.llm_output.append(f"<b>[Download]</b> Skipping {pair} ({tf}) [{exchange_label}] - already downloaded and recent.")
                # Smooth progress bar animation for skipped
                current_val = self.progress_bar.value()
                target_val = current_val + 1
                steps = 10
                for step in range(1, steps + 1):
                    self.progress_bar.setValue(current_val + (step * (target_val - current_val)) // steps)
                    QApplication.processEvents()
                    time.sleep(0.01)
                continue
            self.llm_output.append(f"<b>[Download]</b> Downloading {limit} rows for {pair} ({tf}) from {exchange_label}...")
            QApplication.processEvents()
            ohlcv = fetch_func(pair, tf, limit=limit)
            if ohlcv:
                save_ohlcv(pair, tf, ohlcv)
                self.llm_output.append(f"<b>[Download]</b> Saved {len(ohlcv)} rows for {pair} ({tf}) [{exchange_label}]")
            else:
                self.llm_output.append(f"<b>[Download]</b> Failed to fetch data for {pair} ({tf}) [{exchange_label}]")
            # Smooth progress bar animation
            current_val = self.progress_bar.value()
            target_val = current_val + 1
            steps = 10
            for step in range(1, steps + 1):
                self.progress_bar.setValue(current_val + (step * (target_val - current_val)) // steps)
                QApplication.processEvents()
                time.sleep(0.03)
        self.progress_bar.setVisible(False)
        self.llm_output.append(f"<b>[Download]</b> A month of all timeframes downloaded for {pair} ({exchange_label}).")

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

    def detect_reversal(self, ohlcv: list) -> str:
        """
        Simple reversal detection using last 3 candles (bullish/bearish engulfing).
        Returns: 'bullish', 'bearish', or 'none'
        """
        if len(ohlcv) < 3:
            return 'none'
        prev2, prev1, last = ohlcv[-3], ohlcv[-2], ohlcv[-1]
        # Bullish engulfing: last close > last open, last open < prev1 close, last close > prev1 open
        if last[4] > last[1] and last[1] < prev1[4] and last[4] > prev1[1]:
            return 'bullish'
        # Bearish engulfing: last close < last open, last open > prev1 close, last close < prev1 open
        if last[4] < last[1] and last[1] > prev1[4] and last[4] < prev1[1]:
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
        """
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
            custom = self.select_pair_button.text().strip().upper()
            if custom:
                symbol = f"{custom}/USDT"
            else:
                self.llm_output.append("<b>[ML Training]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
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
                self.llm_output.setHtml("<b>[ML Training]</b> No historical data available for this symbol/timeframe.")
                after_analysis()
                return
            valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
            if valid_rows < indicator_lookback:
                self.llm_output.setHtml(
                    f"<b>[ML Training]</b> Not enough valid data for indicator calculation. At least {indicator_lookback} rows are required after cleaning. Valid rows: {valid_rows}"
                )
                after_analysis()
                return
            usable_rows = min(rows, valid_rows - indicator_lookback)
            if usable_rows < 1:
                self.llm_output.setHtml(
                    f"<b>[ML Training]</b> Not enough data after indicator lookback. Try lowering the rows setting or using a higher timeframe."
                )
                after_analysis()
                return
            if usable_rows < rows:
                self.llm_output.append(
                    f"<b>[ML Training]</b> Only {usable_rows} rows available for training after indicator calculation. Proceeding with available data."
                )
            try:
                df_hist = self.add_technical_indicators(df_hist, min_rows=usable_rows)
            except ValueError as e:
                self.llm_output.setHtml(f"<b>[ML Training]</b> {e}")
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
                                self.llm_output.append(f"<b>[ML Training]</b> Hyperparameter-tuned model saved to {model_path_optuna}")
                            else:
                                self.llm_output.append("<b>[ML Training]</b> Warning: No pipeline_ attribute found after Optuna tuning.")
                        except Exception as e:
                            self.llm_output.append(f"<b>[ML Training]</b> Error saving Optuna-tuned model: {e}")
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
                        self.llm_output.setHtml(html)
                        after_training()
                    def on_error(msg):
                        progress.close()
                        self.llm_output.setHtml(f"<b>[ML Training]</b> Error during tuning: {msg}")
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
                    self.llm_output.append(f"<b>[ML Training]</b> Regular model saved to {model_path}")
                else:
                    self.llm_output.append("<b>[ML Training]</b> Warning: No pipeline_ attribute found after training.")
            except Exception as e:
                self.llm_output.append(f"<b>[ML Training]</b> Error saving regular model: {e}")
            self.llm_output.setHtml(self.format_ml_training_result_html(results))
            after_training()
        patched_train_model()

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

# --- Ollama health check ---
def ollama_is_running() -> bool:
    """Check if Ollama is running locally by pinging the /api/tags endpoint."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

class TickerFetcher(QThread):
    data_fetched = pyqtSignal(dict)
    def __init__(self, get_symbol_func):
        super().__init__()
        self.get_symbol_func = get_symbol_func
        self._running = True
    def run(self):
        import time
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 