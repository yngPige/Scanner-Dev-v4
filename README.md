# 3lacks Trading Terminal

## Overview

**3lacks Trading Terminal** is a professional-grade, modular, and extensible cryptocurrency analytics and trading research platform. It provides a powerful PyQt6 GUI for real-time and historical data analysis, machine learning (ML) and LLM-powered market insights, and advanced backtesting. The platform is designed for both traders and developers, with a focus on transparency, reproducibility, and extensibility.

---

## Key Features

- **Modern PyQt6 GUI**: Intuitive, responsive interface for all trading analytics, with real-time updates and advanced controls.
- **Multi-Exchange Data**: Fetches OHLCV and ticker data from OKX and multiple CCXT-supported exchanges (BinanceUS, Kraken, Bybit, Kucoin, etc.).
- **Machine Learning & LSTM**: Built-in ML models (RandomForest, GradientBoosting, SVM, XGBoost, LSTM) for market prediction, with user-configurable training, Optuna hyperparameter tuning, and walk-forward backtesting.
- **LLM Integration**: Local LLM (Ollama) for technical analysis summaries, using context-rich prompts (symbol, timeframe, data range, ML metrics).
- **Advanced Technical Indicators**: Computes dozens of indicators using `pandas-ta` (with default column names for ML compatibility).
- **Comprehensive Reporting**: HTML-formatted analysis, feature importance charts, and export options (CSV, PDF planned).
- **LSTM Stream Output**: Real-time, color-coded table and sparkline chart for LSTM predictions, with dark mode and resource usage display.
- **Batch Download & Screener**: Download all pairs/timeframes, run fast ML screening, and filter results interactively.
- **Robust Error Handling**: User-friendly dialogs and feedback for missing/insufficient data, model issues, and more.
- **Extensible & Modular**: Clean architecture for adding new exchanges, models, analytics, or UI features.

---

## Architecture

- **GUI Layer**: `Features/gui/main_window.py` (main window, controls, data table, output window, dialogs)
- **Data Layer**: `Data/` (exchange fetchers, data storage)
- **ML/Analysis Layer**: `src/analysis/`, `src/utils/` (MLAnalyzer, LSTM, helpers, technical indicators)
- **Backtesting**: `backtest_strategies/` (strategy implementations)
- **Configuration**: `config.py`, `LLMConfig.py`
- **Testing**: `tests/` (pytest-based unit and integration tests)

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yngPige/Scanner-Dev-v4.git
cd "Scanner Dev v4"
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install pyqtgraph
```

### 4. (Optional) Install Ollama for LLM Analysis

- Download and run Ollama from [https://ollama.com/](https://ollama.com/)
- Ensure it is running locally at `http://localhost:11434`

---

## Usage

### Launch the GUI

```bash
python main.py
```

- **Pair Selection**: Use the "Select Pair" button to choose from all available pairs across supported exchanges.
- **Controls**: Set timeframe, ML algorithm, analysis mode, and Ollama model. Use the main control buttons to fetch data, analyze, train, backtest, and more.
- **Data Table**: Displays the latest OHLCV data for the selected pair and timeframe.
- **Output Window**: Shows logs, ML/LLM analysis, and results. Located directly below the data table for easy reference.
- **LSTM Stream Output**: Click "Show LSTM Stream Output" for a real-time table and chart of LSTM predictions, with resource usage and settings.
- **Feature Report**: View feature importances and sample values used in ML models.
- **Screener**: Run a fast ML screener across all pairs and filter results interactively.

### Run Tests

```bash
pytest tests/
```

---

## UI Layout

- **Top Controls**: Pair selection, timeframe, ML algorithm, analysis mode, Ollama model, strategy, and main action buttons (Fetch, Analyze, Train, Run, Backtest, etc.).
- **Data Table**: Shows recent OHLCV data, color-coded for bullish/bearish moves.
- **Output Window**: (QTextEdit) Directly below the data table, displays logs, analysis results, and LLM output.
- **Status & Indicators**: Current price, reversal indicator, last data fetch timestamp, and LSTM forecast.
- **LSTM Stream Output**: Separate dialog with real-time table, sparkline, dark mode, and resource usage.
- **Dialogs**: For feature reports, backtest results, screener, and more.

---

## Configuration

- **ML/LLM Settings**: See `config.py` and `LLMConfig.py` for algorithm and model settings.
- **Adding Exchanges/Models**: Extend `Data/`, `src/analysis/`, or `src/utils/` as needed. All new features should follow the modular design and type annotation guidelines.

---

## Troubleshooting & Common Issues

- **Missing Features/Indicators**: All technical indicators use pandas_ta default column names. Ensure data preparation is consistent for both training and prediction.
- **ModuleNotFoundError**: Run the GUI from the project root, and ensure all required folders are present and in your Python path.
- **LLM Not Running**: Start Ollama locally for LLM analysis features.
- **Not Enough Data**: Minimum row count for ML training is user-configurable. Lower the minimum or fetch more data if needed.
- **PyQtGraph Missing**: Install with `pip install pyqtgraph` if you see import errors.

---

## Contribution Guidelines

- **Code Style**: Follow PEP 8 and use `ruff` for linting/formatting.
- **Type Annotations**: All functions/classes must have strict type annotations (`typing` module).
- **Docstrings**: Use Google-style docstrings for all public functions/classes.
- **Testing**: Add/modify tests in `tests/` for new features. Aim for 90%+ coverage.
- **Commits**: Use clear, descriptive commit messages.
- **Security**: Never commit API keys, `.env`, or sensitive data.
- **Pull Requests**: Open PRs for all changes. Include a summary and reference related issues.

---

## Security & Best Practices

- Use virtual environments for isolation.
- Pin all dependencies for reproducibility.
- Review code for security, especially when handling user input or external data.
- Never commit sensitive files (e.g., `.env`, API keys, large data files).

---

## License

MIT License (or specify your license here)

---

## Acknowledgements

- [PyQt6](https://riverbankcomputing.com/software/pyqt/intro/)
- [ccxt](https://github.com/ccxt/ccxt)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)
- [Ollama](https://ollama.com/)
- [scikit-learn](https://scikit-learn.org/)

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the maintainer.

---

**Tip:** For advanced usage, see the docstrings in `Features/gui/main_window.py` and the `tests/` directory for examples of programmatic access and extension. 
