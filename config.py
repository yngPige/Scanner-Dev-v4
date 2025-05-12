"""
Configuration for MLAnalyzer and trading parameters.
"""
from typing import Dict, Any
import os
from dotenv import load_dotenv
from Data.blofin_oublic_data import fetch_blofin_instruments
load_dotenv()  # This loads variables from .env into os.environ

api_key = os.environ.get("BLOFIN_API_KEY")
secret = os.environ.get("BLOFIN_SECRET")
passphrase = os.environ.get("BLOFIN_PASSPHRASE")
markets = fetch_blofin_instruments(api_key, secret, passphrase)

# ML Algorithm selection
ML_ALGORITHM: str = "RandomForest"  # Options: "RandomForest", "GradientBoosting", "SVM"
ML_HYPERPARAMETER_TUNING: bool = True
ML_TRAINING_LOOKBACK: int = 500  # Number of rows to use for training
ML_PREDICTION_HORIZON: int = 3   # How many periods ahead to predict
ML_CONFIDENCE_THRESHOLD: float = 50.0  # Lowered for more actionable signals
ML_FEATURE_IMPORTANCE_THRESHOLD: float = 0.01  # Minimum importance to consider a feature
DEFAULT_TIMEFRAME: str = "1h"

# RandomForest parameters
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_SPLIT: int = 2
RF_MIN_SAMPLES_LEAF: int = 1
RF_RANDOM_STATE: int = 42

# RandomForest hyperparameter grid
RF_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Trading thresholds
STOP_LOSS_PERCENTAGE: float = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE: float = 0.04  # 4% take profit

# Add more config as needed for other algorithms or features from dotenv import load_dotenv
