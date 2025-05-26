# 3lacks Scanner

## Overview

**3lacks Scanner** is a modern, modular, and extensible trading analytics platform and GUI terminal for cryptocurrency markets. It combines:
- Real-time and historical data fetching (OKX exchange)
- Machine Learning (ML) market analysis and backtesting
- LLM-powered trading signal generation (Ollama integration)
- Interactive, user-friendly GUI (PyQt6)
- Advanced technical indicators (pandas-ta, with default column names for ML consistency)
- Modular architecture for easy extension

---

## Features

- **GUI Trading Terminal**: Intuitive PyQt6 interface for data exploration, ML/LLM analysis, and reporting.
- **Exchange Integration**: Fetches OHLCV and ticker data from OKX using `ccxt`.
- **Machine Learning**: Built-in ML models (RandomForest, GradientBoosting, SVM, XGBoost) for market prediction, with training and backtesting.
- **User-Configurable Training**: Set the minimum number of valid rows for ML training via a dialog (default: 35, range: 10-500).
- **Technical Indicators**: Computes dozens of indicators using `pandas-ta` (all features use pandas_ta default column names for model compatibility).
- **Crypto Price Formatting**: All crypto prices (open, high, low, close, support/resistance, etc.) are displayed with 4 decimal places for clarity and consistency.
- **LLM Analysis**: Integrates with local Ollama LLMs for technical summaries (prompt includes symbol, timeframe, and data range context).
- **Historical Data Management**: Download, store, and analyze historical OHLCV data.
- **Export & Reporting**: Export analysis and reports (CSV, PDF planned).
- **Batch Download & Backtesting**: Download bulk data and run walk-forward ML backtests.
- **Comprehensive ML Analysis Output**: ML results are formatted as a detailed, visually clear HTML summary (tables, color highlights, sections).
- **Improved Error Handling**: User-friendly error dialogs and feedback for missing/insufficient data.
- **Extensible**: Modular codebase for adding new exchanges, models, or analytics.
- **LSTM Stream Output**:
  - Enhanced, color-coded real-time table (forecast, delta, percent change, uncertainty, trend)
  - Real-time sparkline chart (forecast, previous close, uncertainty band)
  - Dark mode toggle for the chart
  - Resource usage display (CPU, RAM, GPU)

---

## Installation & Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/yngPige/Scanner-Dev-v4.git
cd "Scanner Dev v4"
```

### 2. **Create and Activate a Virtual Environment**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install pyqtgraph
```

### 4. **(Optional) Install Ollama for LLM Analysis**
- Download and run Ollama from [https://ollama.com/](https://ollama.com/)
- Ensure it is running locally at `http://localhost:11434`

---

## Usage

### **Launch the GUI**
```bash
python Features/3lack_gui.py
```

- **Data Fetching**: Select exchange, pair, and timeframe. Fetch real-time or historical data.
- **ML Analysis**: Choose ML algorithm, set minimum row count, train models, and analyze market data. Results are shown in a comprehensive HTML summary (with tables, color, and sections).
- **LLM Analysis**: Select LLM model (Ollama), generate technical summaries. LLM prompt includes symbol, timeframe, and data range for context.
- **Backtesting**: Run walk-forward ML backtests.
- **Export**: Export analysis and reports (CSV, PDF planned).

### **Run Tests**
```bash
pytest tests/
```

### **LSTM Stream Output Usage**
- Click **"Show LSTM Stream Output"** in the main window.
- The output window displays:
  - A real-time, color-coded table of LSTM predictions.
  - A sparkline chart showing forecast, previous close, and uncertainty.
  - A dark mode toggle for the chart ("Enable Dark Mode").
  - Resource usage (CPU, RAM, GPU) at the bottom.
- Use the **Settings** tab to adjust streaming interval, window size, and uncertainty iterations.
- All features/indicators are always enabled and listed in the settings tab.

## Notes

- Ensure you have a compatible GPU and PyTorch installed for best LSTM performance.
- If you encounter `ModuleNotFoundError: No module named 'pyqtgraph'`, install it as shown above.
- For best results, train the LSTM model before starting the stream.

---

## Git & Version Control Best Practices

- **.gitignore**: Ensure your `.gitignore` includes:
  ```
  __pycache__/
  .venv/
  *.pyc
  .env
  dist/
  build/
  Data/
  ```
- **Pushing to GitHub**:
  ```bash
  git add .
  git commit -m "Your commit message"
  git push origin main
  # Or, for first push:
  git branch -M main
  git remote add origin git@github.com:yngPige/Dev.git
  git push -u origin main
  ```
- **Never commit sensitive files** (e.g., `.env`, API keys, large data files).

---

## Dependencies

All dependencies are pinned in `requirements.txt`. Key packages:
- **PyQt6**: GUI framework
- **pandas, numpy, scipy**: Data analysis
- **scikit-learn, joblib**: Machine learning
- **ccxt**: Crypto exchange API
- **pandas-ta**: Technical indicators (default column names enforced)
- **requests**: HTTP requests
- **pytest**: Testing
- **Ollama**: (external, for LLM analysis)

---

## Configuration

- **ML/LLM Settings**: See `config.py` and `LLMConfig.py` for algorithm and model settings.
- **Add new exchanges or models**: Extend `Data/`, `src/analysis/`, or `src/utils/` as needed.

---

## Troubleshooting & Common Issues

- **Feature Names Seen at Fit Time, Yet Now Missing**: All technical indicators use pandas_ta default column names. If you see this error, ensure your data preparation uses the same indicator naming for both training and prediction.
- **ModuleNotFoundError: No module named 'Data'**: Make sure you run the GUI from the project root, and that the `Data` folder is present and in your Python path.
- **pandas fillna Deprecation Warning**: The code now uses `df.ffill().bfill()` instead of the deprecated `df.fillna(method=...)`.
- **LLM Output is Too Verbose**: The LLM prompt is now engineered to provide concise technical summaries, not trading recommendations.
- **Not Enough Data for Training**: The minimum row count for ML training is user-configurable. If you see this error, try lowering the minimum or fetching more data.

---

## Contribution Guidelines

- Follow PEP 8 and use `ruff` for linting/formatting.
- All functions/classes must have type annotations and Google-style docstrings.
- Add/modify tests in `tests/` for new features.
- Use clear, descriptive commit messages.
- Open issues or pull requests for bugs, features, or questions.

---

## Security & Best Practices

- Never commit API keys or sensitive data.
- Use virtual environments for isolation.
- Pin all dependencies for reproducibility.
- Review code for security, especially when handling user input or external data.

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
