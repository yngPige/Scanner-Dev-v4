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
    QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QAbstractItemView, QDoubleSpinBox, QSpinBox, QInputDialog, QMessageBox, QProgressBar, QCompleter, QDialog, QListWidget, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon
from datetime import datetime, timezone
from Data.okx_public_data import fetch_okx_ticker
from Data.blofin_oublic_data import fetch_blofin_instruments, get_blofin_spot_pairs_via_ws
from backtest_strategies.strategies import moving_average_crossover_strategy, rsi_strategy, buy_and_hold_strategy
from src.utils.data_storage import save_ohlcv, load_ohlcv
from src.utils.helpers import fetch_ohlcv
import pandas as pd
import pandas_ta as ta
import numpy as np
from dotenv import load_dotenv
import os
import requests
from src.analysis.ml import MLAnalyzer, launch_optuna_dashboard

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
        self.setWindowIcon(QIcon("assets/icon.png"))
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
        self.algorithm_combo.addItems(["RandomForest", "GradientBoosting", "SVM", "XGBoost"])
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

        # Test API Button (was OKX API Test Button)
        self.test_api_button = QPushButton("Test API")
        self.test_api_button.clicked.connect(self.run_api_test)

        # Download Button
        self.download_button = QPushButton("Download Historical Data")
        self.download_button.clicked.connect(self.download_historical_data)

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

        # Add Optuna Dashboard Button
        self.optuna_dashboard_button = QPushButton("Optuna Dashboard")
        self.optuna_dashboard_button.clicked.connect(self.run_optuna_dashboard)

        # Add Current section for real-time data
        self.current_label = QLabel("<b>Current:</b> --")
        self.current_label.setTextFormat(Qt.TextFormat.RichText)

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
        main_layout.addWidget(self.download_button)
        main_layout.addWidget(self.download_all_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.backtest_button)
        main_layout.addWidget(self.optuna_dashboard_button)
        main_layout.addWidget(self.test_api_button)
        main_layout.addWidget(self.current_label)
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

    def fetch_data(self):
        self.update_exchange_status()
        symbol = self.select_pair_button.text().strip().upper()
        if not symbol or symbol == "SELECT PAIR":
            self.data_table.setRowCount(0)
            self.data_table.setVisible(False)
            self.llm_output.setText("Please select a pair.")
            return
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        rows = 1000
        exchange = getattr(self, 'selected_exchange', 'OKX')
        use_ws = (exchange == "Blofin")
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
        else:
            self.data_table.setRowCount(0)
            self.data_table.setVisible(False)
            self.llm_output.setText("Failed to fetch OHLCV data. Please check the symbol, timeframe, or your internet connection.")

    def run_analysis(self):
        """Run analysis using MLAnalyzer only."""
        mode = self.analysis_mode_combo.currentText()
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}"
        else:
            self.llm_output.setHtml("<b>[Analysis]</b> Please select an exchange and pair.")
            return
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
            self.llm_output.setHtml("<b>[Analysis]</b> No historical data available for this symbol/timeframe.")
            return
        try:
            df_hist = self.add_technical_indicators(df_hist)
        except ValueError as e:
            self.llm_output.setHtml(f"<b>[Analysis]</b> {e}")
            return
        # Only use MLAnalyzer
        self.ml_analyzer.set_custom_model_class(None)
        output = ""
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
                # Only pass the original OHLCV columns
                ohlcv_data = df_hist[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
                llm_result = self.llm_analyze_ohlcv(ohlcv_data)
                output += "[LLM Analysis]\n" + llm_result
        self.llm_output.setHtml(output)

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
        html += f"<li><b>Position Size:</b> {result.get('metrics',{}).get('position_size_recommendation',0):.2f}%</li>"
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
        import time
        import logging
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame is empty or missing required OHLCV columns.")
        if df['close'].isnull().all():
            raise ValueError("No valid 'close' price data available for indicator calculation.")
        timings = {}
        t0 = time.perf_counter()
        df = df.rename(columns={col: col.lower() for col in df.columns})
        timings['rename'] = time.perf_counter() - t0
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}. "
                f"Found: {set(df.columns)}"
            )
        # Add all indicators with pandas_ta default names
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
            t0 = time.perf_counter()
            if ind is not None:
                df = pd.concat([df, ind], axis=1)
            timings[str(ind)] = time.perf_counter() - t0
        df = df.dropna().reset_index(drop=True)
        logging.info("[PROFILE] Indicator timings: %s", timings)
        print("[PROFILE] Indicator timings:", timings)
        # Ensure at least min_rows after cleaning
        valid_rows = len(df.dropna(subset=["open", "high", "low", "close", "volume"]))
        if valid_rows < min_rows:
            raise ValueError(f"Not enough valid data for indicator calculation. At least {min_rows} rows are required after cleaning. Valid rows: {valid_rows}")
        return df

    def train_model(self):
        # Use a default or dynamically determined rows value
        rows = 1000  # Default value, or compute based on timeframe/algorithm if needed
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}/USDT"
        else:
            self.llm_output.append("<b>[ML Training]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
            return
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        # Determine the max lookback needed by indicators (e.g., EMA 100, MACD 26, etc.)
        indicator_lookback = max([100, 26, 20, 14])  # EMA 100, MACD slow, BBands, ATR, etc.
        min_required_rows = rows + indicator_lookback
        df_hist = load_ohlcv(symbol, norm_timeframe)
        # If not enough rows, try to fetch more
        if df_hist.shape[0] < min_required_rows:
            from Data.okx_public_data import fetch_okx_ohlcv
            ohlcv = fetch_okx_ohlcv(symbol, norm_timeframe, limit=min_required_rows)
            if ohlcv:
                save_ohlcv(symbol, norm_timeframe, ohlcv)
                df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
            self.llm_output.setHtml("<b>[ML Training]</b> No historical data available for this symbol/timeframe.")
            return
        valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
        # Dynamically adjust rows to fit indicator requirements
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
        results = self.ml_analyzer.train_model(df_hist)
        self.llm_output.setHtml(self.format_ml_training_result_html(results))

    def format_ml_training_result_html(self, result: dict) -> str:
        """
        Format the ML training result as a human-friendly HTML summary for display, with color-coded metrics.
        """
        if not isinstance(result, dict):
            return f"<b>[ML Training]</b> Invalid result."
        if 'error' in result:
            return f"<b>[ML Training Error]</b> {result['error']}"

        def color_metric(val: float) -> str:
            if val >= 0.8:
                return 'green'
            elif val >= 0.6:
                return 'gold'
            else:
                return 'red'
        def color_sharpe(val: float) -> str:
            if val >= 1.0:
                return 'green'
            elif val >= 0.5:
                return 'gold'
            else:
                return 'red'

        accuracy = result.get('accuracy', 0)
        precision = result.get('precision', 0)
        recall = result.get('recall', 0)
        f1 = result.get('f1_score', 0)
        sharpe = result.get('mean_sharpe', result.get('sharpe', 0))

        html = f"""
        <div style='font-family:Arial,sans-serif;'>
        <h2 style='margin-bottom:0;'>ML Model Training Summary</h2>
        <b>Algorithm:</b> {result.get('algorithm', '')}<br>
        <b>Accuracy:</b> <span style='color:{color_metric(accuracy)};font-weight:bold;'>{accuracy:.4f}</span><br>
        <b>Precision:</b> <span style='color:{color_metric(precision)};font-weight:bold;'>{precision:.4f}</span><br>
        <b>Recall:</b> <span style='color:{color_metric(recall)};font-weight:bold;'>{recall:.4f}</span><br>
        <b>F1 Score:</b> <span style='color:{color_metric(f1)};font-weight:bold;'>{f1:.4f}</span><br>
        <b>Sharpe Ratio:</b> <span style='color:{color_sharpe(sharpe)};font-weight:bold;'>{sharpe:.4f}</span><br>
        <b>Training Time:</b> {result.get('training_time', 0):.2f} seconds<br>
        <b>Timestamp:</b> {result.get('timestamp', '')}<br>
        <hr style='margin:8px 0;'>
        </div>
        """
        return html

    def download_historical_data(self):
        rows = 1000
        custom = self.select_pair_button.text().strip().upper()
        if not custom:
            self.llm_output.setHtml("<b>[Download]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
            return
        symbol = f"{custom}/USDT"
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe).lower()  # Normalize to lowercase for CCXT
        # Add CCXT exchanges to the dialog, including ALL PAIRS options
        ccxt_exchanges = [f"CCXT:{exch}" for exch in ["binance", "okx", "kraken", "bybit", "kucoin"]]
        ccxt_all_pairs = [f"CCXT:{exch} (ALL PAIRS)" for exch in ["binance", "okx", "kraken", "bybit", "kucoin"]]
        exchanges = ["OKX", "Blofin", "MEXC"] + ccxt_exchanges + ccxt_all_pairs + ["Both"]
        exchange, ok = QInputDialog.getItem(self, "Select Exchange", "Choose exchange to download from:", exchanges, 0, False)
        if not ok or not exchange:
            self.llm_output.append("<b>[Download]</b> Download cancelled by user.")
            return
        self.progress_bar.setMaximum(2)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat(f"Downloading {rows} rows of historical data... %p%")
        QApplication.processEvents()
        results = []
        # Use generic fetch_ohlcv for all exchanges
        exch_list = [ex for ex in exchanges if ex != "Both"] if exchange == "Both" else [exchange]
        for exch in exch_list:
            use_ws = (exch == "Blofin")
            # Handle CCXT ALL PAIRS batch download
            if exch.startswith("CCXT:") and "(ALL PAIRS)" in exch:
                ccxt_name = exch.split(":", 1)[1].split()[0]
                from Data.ccxt_public_data import fetch_ccxt_ohlcv_all_pairs
                all_ohlcv = fetch_ccxt_ohlcv_all_pairs(ccxt_name, norm_timeframe, rows)
                total_pairs = len(all_ohlcv)
                self.progress_bar.setMaximum(total_pairs)
                for idx, (pair, ohlcv) in enumerate(all_ohlcv.items(), 1):
                    if ohlcv:
                        save_ohlcv(pair, norm_timeframe, ohlcv)
                        self.llm_output.append(f"<b>[Download]</b> Downloaded {len(ohlcv)} rows for {pair} ({norm_timeframe}) from {exch}.")
                        results.append((exch, ohlcv))
                    else:
                        self.llm_output.append(f"<b>[Download]</b> Failed to download historical data for {pair} ({norm_timeframe}) from {exch}.")
                    self.progress_bar.setValue(idx)
                    QApplication.processEvents()
                continue
            # Handle single CCXT pair
            if exch.startswith("CCXT:"):
                ccxt_name = exch.split(":", 1)[1].split()[0]
                from Data.ccxt_public_data import fetch_ccxt_ohlcv
                ohlcv = fetch_ccxt_ohlcv(ccxt_name, symbol, norm_timeframe, rows)
                if ohlcv:
                    save_ohlcv(symbol, norm_timeframe, ohlcv)
                    self.llm_output.append(f"<b>[Download]</b> Downloaded {len(ohlcv)} rows for {symbol} ({norm_timeframe}) from {exch}. Starting training...")
                    results.append((exch, ohlcv))
                else:
                    self.llm_output.append(f"<b>[Download]</b> Failed to download historical data for {symbol} ({norm_timeframe}) from {exch}.")
                self.progress_bar.setValue(self.progress_bar.value() + 1)
                QApplication.processEvents()
                continue
            ohlcv = fetch_ohlcv(exch, symbol, norm_timeframe, limit=rows, use_ws=use_ws)
            if ohlcv:
                save_ohlcv(symbol, norm_timeframe, ohlcv)
                self.llm_output.append(f"<b>[Download]</b> Downloaded {len(ohlcv)} rows for {symbol} ({norm_timeframe}) from {exch}. Starting training...")
                results.append((exch, ohlcv))
            else:
                self.llm_output.append(f"<b>[Download]</b> Failed to download historical data for {symbol} ({norm_timeframe}) from {exch}.")
            self.progress_bar.setValue(self.progress_bar.value() + 1)
            QApplication.processEvents()
        self.progress_bar.setValue(1)
        QApplication.processEvents()
        for exch, ohlcv in results:
            df_hist = load_ohlcv(symbol, norm_timeframe)
            if df_hist.empty or 'close' not in df_hist or df_hist['close'].isnull().all():
                self.progress_bar.setVisible(False)
                self.llm_output.setHtml(f"<b>[ML Training]</b> No valid historical data available for this symbol/timeframe from {exch}.")
                continue
            df_hist = df_hist.rename(columns={col: col.lower() for col in df_hist.columns})
            valid_rows = len(df_hist.dropna(subset=["open", "high", "low", "close", "volume"]))
            if valid_rows < rows:
                self.llm_output.append(
                    f"<b>[ML Training]</b> Only {valid_rows} rows available for training after indicator calculation from {exch}. Proceeding with available data."
                )
            try:
                df_hist = self.add_technical_indicators(df_hist, min_rows=valid_rows)
            except ValueError as e:
                self.progress_bar.setVisible(False)
                self.llm_output.setHtml(f"<b>[ML Training]</b> {e} ({exch})")
                continue
            self.ml_analyzer.set_custom_model_class(None)
            results_ml = self.ml_analyzer.train_model(df_hist)
            self.llm_output.append(f"<b>[ML Training]</b> Training complete for {exch}.")
            self.llm_output.append(self.format_ml_training_result_html(results_ml))
        self.progress_bar.setValue(2)
        self.progress_bar.setVisible(False)

    def run_backtest(self):
        custom = self.select_pair_button.text().strip().upper()
        if custom:
            symbol = f"{custom}/USDT"
        else:
            self.llm_output.setHtml("<b>[Backtest]</b> Please enter a symbol (e.g., BTC) in the Pair field.")
            return
        timeframe = self.timeframe_combo.currentText()
        norm_timeframe = self.timeframe_map.get(timeframe, timeframe)
        rows = 1000
        exchange = getattr(self, 'selected_exchange', 'OKX')
        use_ws = (exchange == "Blofin")
        ohlcv = fetch_ohlcv(exchange, symbol, norm_timeframe, limit=rows, use_ws=use_ws)
        if ohlcv:
            save_ohlcv(symbol, norm_timeframe, ohlcv)
            df_hist = load_ohlcv(symbol, norm_timeframe)
        else:
            df_hist = load_ohlcv(symbol, norm_timeframe)
        if df_hist.empty:
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
        self.llm_output.setHtml(self.format_backtest_result_html(results, selected_strategy))

    def format_backtest_result_html(self, result: dict, strategy_name: str) -> str:
        html = f"""
        <div style='font-family:Arial,sans-serif;'>
        <h2 style='margin-bottom:0;'>Backtest Summary</h2>
        <b>Strategy:</b> {strategy_name}<br>
        <b>Algorithm:</b> {result.get('algorithm', '')}<br>
        <b>Splits:</b> {result.get('splits', 'N/A')}<br>
        <b>Timestamp:</b> {result.get('timestamp', '')}<br>
        <hr style='margin:8px 0;'>
        <b>Mean Metrics:</b><br>
        <table border='1' cellpadding='4' cellspacing='0'>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td align='right'>{result.get('accuracy', 0):.4f}</td></tr>
            <tr><td>F1 Score</td><td align='right'>{result.get('f1', 0):.4f}</td></tr>
            <tr><td>Precision</td><td align='right'>{result.get('precision', 0):.4f}</td></tr>
            <tr><td>Recall</td><td align='right'>{result.get('recall', 0):.4f}</td></tr>
            <tr><td>Balanced Accuracy</td><td align='right'>{result.get('balanced_accuracy', 0):.4f}</td></tr>
            <tr><td>ROC-AUC</td><td align='right'>{result.get('auc', 0) if result.get('auc') is not None else 'N/A'}</td></tr>
            <tr><td>Cumulative Return</td><td align='right'>{result.get('mean_cumulative_return', 0):.4f}</td></tr>
            <tr><td>Max Drawdown</td><td align='right'>{result.get('mean_max_drawdown', 0):.4f}</td></tr>
            <tr><td>Sharpe Ratio</td><td align='right'>{result.get('mean_sharpe', 0):.4f}</td></tr>
        </table>
        <br>
        <b>Per-Split Metrics:</b><br>
        <table border='1' cellpadding='4' cellspacing='0'>
            <tr><th>Split</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>Balanced Acc</th><th>ROC-AUC</th><th>Return</th><th>Drawdown</th><th>Sharpe</th></tr>
        """
        for i, m in enumerate(result.get('split_metrics', [])):
            html += f"<tr><td>{i+1}</td>"
            html += f"<td align='right'>{m.get('accuracy', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('f1', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('precision', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('recall', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('balanced_accuracy', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('auc', 'N/A') if m.get('auc') is not None else 'N/A'}</td>"
            html += f"<td align='right'>{m.get('cumulative_return', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('max_drawdown', 0):.4f}</td>"
            html += f"<td align='right'>{m.get('sharpe', 0):.4f}</td>"
            html += "</tr>"
        html += "</table><br>"
        # Per-split confusion matrix
        html += "<b>Per-Split Confusion Matrices:</b><br>"
        for i, m in enumerate(result.get('split_metrics', [])):
            cm = m.get('confusion_matrix')
            if cm:
                html += f"<details><summary>Split {i+1} Confusion Matrix</summary>"
                html += "<table border='1' cellpadding='2' cellspacing='0'>"
                html += "<tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>"
                html += f"<tr><td><b>Actual 0</b></td><td align='center'>{cm[0][0]}</td><td align='center'>{cm[0][1]}</td></tr>"
                html += f"<tr><td><b>Actual 1</b></td><td align='center'>{cm[1][0]}</td><td align='center'>{cm[1][1]}</td></tr>"
                html += "</table></details>"
        # Per-split classification report
        html += "<b>Per-Split Classification Reports:</b><br>"
        for i, m in enumerate(result.get('split_metrics', [])):
            cr = m.get('classification_report')
            if cr:
                html += f"<details><summary>Split {i+1} Classification Report</summary>"
                html += "<table border='1' cellpadding='2' cellspacing='0'>"
                html += "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
                for label in ['0', '1']:
                    if label in cr:
                        html += f"<tr><td>{label}</td>"
                        html += f"<td align='right'>{cr[label]['precision']:.4f}</td>"
                        html += f"<td align='right'>{cr[label]['recall']:.4f}</td>"
                        html += f"<td align='right'>{cr[label]['f1-score']:.4f}</td>"
                        html += f"<td align='right'>{cr[label]['support']}</td></tr>"
                html += "</table></details>"
        html += "</div>"
        return html

    def on_algorithm_changed(self, algorithm: str):
        self.selected_algorithm = algorithm
        import config
        config.ML_ALGORITHM = algorithm  # Update global config
        self.ml_analyzer = MLAnalyzer()  # Recreate analyzer with new config

    def get_current_symbol(self) -> str:
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

    def closeEvent(self, event):
        # Ensure the thread is stopped on close
        if hasattr(self, 'ticker_fetcher'):
            self.ticker_fetcher.stop()
            self.ticker_fetcher.wait()
        super().closeEvent(event)

    def get_ccxt_pairs(self, exchange_name: str) -> set:
        if not hasattr(self, 'ccxt_pair_cache'):
            self.ccxt_pair_cache = {}
        if exchange_name in self.ccxt_pair_cache:
            return self.ccxt_pair_cache[exchange_name]
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class()
            markets = exchange.load_markets()
            pairs = set(s for s in markets if s.endswith('/USDT'))
            self.ccxt_pair_cache[exchange_name] = pairs
            return pairs
        except Exception as e:
            error_set = set([f"Error loading CCXT pairs: {e}"])
            self.ccxt_pair_cache[exchange_name] = error_set
            return error_set

    def show_pair_menu(self):
        # Toggle: If the menu is already open, close it
        if hasattr(self, 'pair_menu_widget') and self.pair_menu_widget is not None:
            self.centralWidget().layout().removeWidget(self.pair_menu_widget)
            self.pair_menu_widget.deleteLater()
            self.pair_menu_widget = None
            self.data_table.setVisible(True)
            return

        from Data.okx_public_data import fetch_okx_ticker
        from Data.blofin_oublic_data import fetch_blofin_instruments, get_blofin_spot_pairs_via_ws
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
        self.ccxt_exchanges = ["binance", "okx", "kraken", "bybit", "kucoin"]
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["All", "OKX", "Blofin", "MEXC"] + [f"CCXT:{exch}" for exch in self.ccxt_exchanges])
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
        # Fetch Blofin pairs via WebSocket (preferred)
        try:
            blofin_pairs = get_blofin_spot_pairs_via_ws(timeout=5)
        except Exception as e:
            blofin_pairs = set([f"Error loading Blofin pairs via WebSocket: {e}"])
        # If WebSocket fails, try REST as fallback
        if not blofin_pairs:
            try:
                import os
                api_key = os.environ.get("BLOFIN_API_KEY")
                secret = os.environ.get("BLOFIN_SECRET")
                passphrase = os.environ.get("BLOFIN_PASSPHRASE")
                instruments = fetch_blofin_instruments(api_key, secret, passphrase)
                blofin_pairs = set()
                for inst in instruments or []:
                    if inst.get("quoteCcy", "") == "USDT":
                        base = inst.get("baseCcy")
                        quote = inst.get("quoteCcy")
                        if base and quote:
                            blofin_pairs.add(f"{base}/USDT")
            except Exception as e:
                blofin_pairs = set([f"Error loading Blofin pairs: {e}"])
        # Fetch MEXC pairs
        try:
            import requests
            resp = requests.get("https://api.mexc.com/api/v3/exchangeInfo", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            mexc_pairs = set()
            for s in data.get('symbols', []):
                if s.get('quoteAsset') == 'USDT' and s.get('status') == 'ENABLED':
                    mexc_pairs.add(f"{s['baseAsset']}/USDT")
        except Exception as e:
            mexc_pairs = set([f"Error loading MEXC pairs: {e}"])
        # Merge and mark source (do not fetch CCXT pairs here)
        all_pairs_set = okx_pairs | blofin_pairs | mexc_pairs
        self.pair_sources = {}
        for p in all_pairs_set:
            srcs = []
            if p in okx_pairs:
                srcs.append("OKX")
            if p in blofin_pairs:
                srcs.append("Blofin")
            if p in mexc_pairs:
                srcs.append("MEXC")
            self.pair_sources[p] = srcs
        # Display as: BTC/USDT [OKX,Blofin,MEXC]
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
            # Prefer OKX if available, else Blofin, else MEXC, else CCXT
            if "OKX" in srcs:
                self.selected_exchange = "OKX"
            elif "Blofin" in srcs:
                self.selected_exchange = "Blofin"
            elif "MEXC" in srcs:
                self.selected_exchange = "MEXC"
            elif any(s.startswith("CCXT:") for s in srcs):
                ccxt_src = next(s for s in srcs if s.startswith("CCXT:"))
                self.selected_exchange = ccxt_src.lower()
            else:
                self.selected_exchange = "OKX"
        else:
            pair = label.strip()
            # If the label is from a CCXT dynamic load, set selected_exchange accordingly
            if "[CCXT:" in label:
                ccxt_src = label.split("[CCXT:")[-1].strip("] ")
                self.selected_exchange = f"ccxt:{ccxt_src.lower()}"
            else:
                self.selected_exchange = "OKX"
        self.select_pair_button.setText(pair)
        self.selected_pair = pair
        # Remove the menu and restore the data table
        self.centralWidget().layout().removeWidget(self.pair_menu_widget)
        self.pair_menu_widget.deleteLater()
        self.pair_menu_widget = None  # Ensure reference is cleared
        self.data_table.setVisible(True)
        # Automatically fetch data for the selected pair
        self.fetch_data()

    def download_all_timeframes_for_selected_pair(self):
        """
        Download at least a month (30 days) of OHLCV data for all supported timeframes for the selected pair from both OKX (REST) and Blofin (WebSocket), if available.
        """
        from Data.okx_public_data import fetch_okx_ohlcv
        from Data.blofin_oublic_data import BlofinWebSocketClient
        from src.utils.data_storage import save_ohlcv
        pair = self.select_pair_button.text().strip().upper()
        if not pair or pair == "SELECT PAIR":
            self.llm_output.append("<b>[Download]</b> Please select a pair first.")
            return
        okx_timeframes = ["5m", "15m", "30m", "1h", "4h", "1D"]
        blofin_timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        # Calculate limits for a month (30 days) for each timeframe
        minutes_per_timeframe = {
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1D": 1440,
            "1d": 1440
        }
        # 30 days worth of bars for each timeframe
        bars_per_month = {tf: max(30, int((30*24*60)//minutes_per_timeframe[tf])) for tf in minutes_per_timeframe}
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(okx_timeframes) + len(blofin_timeframes))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Downloading a month of all timeframes (OKX & Blofin)... %p%")
        QApplication.processEvents()
        # Ensure pair_sources is initialized
        if not hasattr(self, 'pair_sources'):
            self.pair_sources = {}
        # Download from OKX if supported
        if "OKX" in self.pair_sources.get(pair, []):
            for tf in okx_timeframes:
                limit = bars_per_month.get(tf, 30)
                self.llm_output.append(f"<b>[Download]</b> Downloading {limit} rows for {pair} ({tf}) from OKX...")
                QApplication.processEvents()
                ohlcv = fetch_okx_ohlcv(pair, tf, limit=limit)
                if ohlcv:
                    save_ohlcv(pair, tf, ohlcv)
                    self.llm_output.append(f"<b>[Download]</b> Saved {len(ohlcv)} rows for {pair} ({tf}) [OKX]")
                else:
                    self.llm_output.append(f"<b>[Download]</b> Failed to fetch data for {pair} ({tf}) [OKX]")
                self.progress_bar.setValue(self.progress_bar.value() + 1)
                QApplication.processEvents()
        # Download from Blofin WebSocket if supported
        if "Blofin" in self.pair_sources.get(pair, []):
            inst_id = pair.replace("/", "-") + "-SPOT" if not pair.endswith("-SPOT") else pair
            ws_client = BlofinWebSocketClient()
            for tf in blofin_timeframes:
                limit = bars_per_month.get(tf, 30)
                self.llm_output.append(f"<b>[Download]</b> Downloading {limit} rows for {pair} ({tf}) via Blofin WebSocket...")
                QApplication.processEvents()
                ohlcv = ws_client.fetch_ohlcv(inst_id, bar=tf, limit=limit, timeout=5.0)
                if ohlcv:
                    save_ohlcv(pair, tf, ohlcv)
                    self.llm_output.append(f"<b>[Download]</b> Saved {len(ohlcv)} rows for {pair} ({tf}) [Blofin]")
                else:
                    self.llm_output.append(f"<b>[Download]</b> Failed to fetch data for {pair} ({tf}) [Blofin]")
                self.progress_bar.setValue(self.progress_bar.value() + 1)
                QApplication.processEvents()
        self.progress_bar.setVisible(False)
        self.llm_output.append(f"<b>[Download]</b> A month of all timeframes downloaded for {pair} (OKX & Blofin).")

    def refresh_data_timestamp(self):
        if self.last_data_fetch_time:
            ts_str = self.last_data_fetch_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            self.data_timestamp_label.setText(f"<b>Last Data Fetch:</b> {ts_str}")
        else:
            self.data_timestamp_label.setText("<b>Last Data Fetch:</b> --")

    def run_optuna_dashboard(self):
        """Prompt for Optuna storage URI and port, then launch the dashboard."""
        storage, ok = QInputDialog.getText(self, "Optuna Storage URI", "Enter Optuna storage URI:", QLineEdit.EchoMode.Normal, "sqlite:///example.db")
        if not ok or not storage:
            return
        port, ok = QInputDialog.getInt(self, "Optuna Dashboard Port", "Enter port for dashboard:", 8080, 1024, 65535, 1)
        if not ok:
            return
        launch_optuna_dashboard(storage, port)
        self.llm_output.append(f"<b>[Optuna]</b> Dashboard launched at <a href='http://localhost:{port}'>http://localhost:{port}</a> for storage: {storage}")

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
    app.setWindowIcon(QIcon("assets/icon.png"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 