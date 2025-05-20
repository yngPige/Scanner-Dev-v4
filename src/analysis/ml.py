"""
Machine Learning-based market analysis for trading decisions
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import config
from src.utils.helpers import calculate_technical_indicators
from src.utils.helpers import compute_support_levels, compute_resistance_levels
from xgboost import XGBClassifier
import warnings
import optuna
from optuna import Trial, Study
from typing import Any, Dict, Optional, Callable
import matplotlib.pyplot as plt
import subprocess
import itertools

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class TechnicalIndicatorAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add technical indicators using calculate_technical_indicators.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is a DataFrame with OHLCV columns
        return calculate_technical_indicators(X)

class MLAnalyzer:
    """Uses machine learning to analyze market data and generate trading signals (with sklearn Pipeline)"""

    def __init__(self):
        """Initialize the ML analyzer based on the configured algorithm"""
        self.algorithm = config.ML_ALGORITHM
        self.hyperparameter_tuning = config.ML_HYPERPARAMETER_TUNING
        self.training_lookback = config.ML_TRAINING_LOOKBACK
        self.prediction_horizon = config.ML_PREDICTION_HORIZON
        self.confidence_threshold = config.ML_CONFIDENCE_THRESHOLD
        self.feature_importance_threshold = config.ML_FEATURE_IMPORTANCE_THRESHOLD
        self.timeframe = config.DEFAULT_TIMEFRAME
        self.ticker_data = None  # Store current ticker data

        # Initialize model directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importances = {}
        self.custom_model_class = None  # For dynamic model selection

        # Load model if it exists
        self._load_model()

    def _load_model(self):
        """Load a saved model if it exists"""
        model_path = os.path.join(self.model_dir, f"{self.algorithm.lower()}_pipeline.joblib")
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Loaded existing {self.algorithm} pipeline model")
                return True
            except Exception as e:
                print(f"Error loading pipeline model: {e}")
                return False
        return False

    def _save_model(self):
        """Save the current model"""
        if self.model is not None:
            # Make sure the model directory exists
            os.makedirs(self.model_dir, exist_ok=True)

            model_path = os.path.join(self.model_dir, f"{self.algorithm.lower()}_pipeline.joblib")

            # Save model
            try:
                joblib.dump(self.model, model_path)
                print(f"Saved {self.algorithm} pipeline model")
            except Exception as e:
                print(f"Error saving pipeline model: {e}")
                return False

            return True
        return False

    def set_custom_model_class(self, model_class: type) -> None:
        """
        Set a custom model class to be used for training and prediction.
        Args:
            model_class (type): The model class to use.
        """
        self.custom_model_class = model_class

    def _create_model(self):
        """Create a new model based on the configured algorithm or custom model class."""
        if self.custom_model_class is not None:
            return self.custom_model_class()
        if self.algorithm == "RandomForest":
            return RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                random_state=config.RF_RANDOM_STATE,
                n_jobs=-1  # Use all available cores
            )
        elif self.algorithm == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.algorithm == "SVM":
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.algorithm == "XGBoost":
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        elif self.algorithm in ["Ensemble (Voting)", "Ensemble (Stacking)"]:
            # Build all base models, skip any that fail
            base_models = []
            errors = []
            # Try to instantiate each model, skip if fails
            try:
                base_models.append(('rf', RandomForestClassifier(
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH,
                    min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                    random_state=config.RF_RANDOM_STATE,
                    n_jobs=-1)))
            except Exception as e:
                errors.append(f"RandomForest: {e}")
            try:
                base_models.append(('gb', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42)))
            except Exception as e:
                errors.append(f"GradientBoosting: {e}")
            try:
                base_models.append(('svm', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42)))
            except Exception as e:
                errors.append(f"SVM: {e}")
            try:
                base_models.append(('xgb', XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1)))
            except Exception as e:
                errors.append(f"XGBoost: {e}")
            # Remove any models that failed
            base_models = [(name, model) for name, model in base_models if model is not None]
            if not base_models:
                raise RuntimeError(f"No base models available for ensemble. Errors: {errors}")
            if self.algorithm == "Ensemble (Voting)":
                return VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
            else:
                # Stacking: use LogisticRegression as final estimator
                from sklearn.linear_model import LogisticRegression
                return StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), n_jobs=-1)
        else:
            # Default to RandomForest
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

    def _build_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', self._create_model())
        ])

    def _prepare_features(self, df):
        """
        Prepare features for the model from technical indicators

        Args:
            df: DataFrame with technical indicators

        Returns:
            X_df: Feature DataFrame (not numpy array)
            feature_names: List of feature names
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Calculate technical indicators if not already present
        if 'rsi' not in df.columns:
            df = calculate_technical_indicators(df)

        # Drop rows with NaN values
        df = df.dropna()

        # Remove future_price and price_direction if they exist
        if 'future_price' in df.columns:
            df = df.drop('future_price', axis=1)
        if 'price_direction' in df.columns:
            df = df.drop('price_direction', axis=1)

        # Select features (exclude timestamp, open, high, low, close, volume)
        exclude_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Create feature DataFrame
        X_df = df[feature_columns]

        return X_df, feature_columns

    def _prepare_labels(self, df, horizon=3):
        """
        Prepare target labels for the model

        Args:
            df: DataFrame with price data
            horizon: Number of periods to look ahead for price movement

        Returns:
            y: Target labels (1 for up, 0 for down)
            df: Updated DataFrame with future_price and price_direction columns
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Calculate future price movement
        df['future_price'] = df['close'].shift(-horizon)
        df['price_direction'] = (df['future_price'] > df['close']).astype(int)

        # Create target vector (before dropping NaN rows)
        y = df['price_direction'].values

        # Note: We don't drop NaN rows here to keep the DataFrame aligned with features
        # The NaN handling will be done in the training/prediction methods

        return y, df

    def train_model(self, klines_data):
        """
        Train the pipeline model on historical data.
        Args:
            klines_data: DataFrame with historical price data
        Returns:
            Dictionary with training results
        """
        start_time = time.time()
        # Step 1: Feature engineering and cleaning BEFORE split
        klines_data = calculate_technical_indicators(klines_data)
        y, labeled_df = self._prepare_labels(klines_data, horizon=self.prediction_horizon)
        y_series = pd.Series(y, index=labeled_df.index)
        # Drop rows with any NaN in features or labels
        mask = ~klines_data.isnull().any(axis=1) & ~y_series.isnull()
        X_clean = klines_data[mask]
        y_clean = y_series[mask]
        if len(X_clean) == 0 or len(y_clean) == 0:
            return {
                'error': 'Not enough data for training',
                'timestamp': datetime.now().isoformat()
            }
        # Step 2: Build pipeline (no indicator adder, no imputer)
        if self.algorithm in ["Ensemble (Voting)", "Ensemble (Stacking)"]:
            # Build all base models, skip any that fail
            base_models = []
            errors = []
            try:
                base_models.append(('rf', RandomForestClassifier(
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH,
                    min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                    random_state=config.RF_RANDOM_STATE,
                    n_jobs=-1)))
            except Exception as e:
                errors.append(f"RandomForest: {e}")
            try:
                base_models.append(('gb', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42)))
            except Exception as e:
                errors.append(f"GradientBoosting: {e}")
            try:
                base_models.append(('svm', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42)))
            except Exception as e:
                errors.append(f"SVM: {e}")
            try:
                base_models.append(('xgb', XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1)))
            except Exception as e:
                errors.append(f"XGBoost: {e}")
            # Remove any models that failed
            base_models = [(name, model) for name, model in base_models if model is not None]
            if not base_models:
                return {
                    'error': f'No base models available for ensemble. Errors: {errors}',
                    'timestamp': datetime.now().isoformat()
                }
            if self.algorithm == "Ensemble (Voting)":
                model = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
            else:
                from sklearn.linear_model import LogisticRegression
                model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), n_jobs=-1)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self._create_model())
            ])
        # Step 3: Split and fit
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        # Feature importances if available
        model_step = pipeline.named_steps['model']
        if hasattr(model_step, 'feature_importances_'):
            feature_names = X_clean.columns.tolist()
            importances = model_step.feature_importances_
            self.feature_importances = dict(zip(feature_names, importances))
        self._save_model()
        training_time = time.time() - start_time
        return {
            'algorithm': self.algorithm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'feature_importances': self.feature_importances,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }

    def predict(self, klines_data):
        """
        Make predictions on new data using the pipeline.
        Args:
            klines_data: DataFrame with price data
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not trained',
                'timestamp': datetime.now().isoformat()
            }
        # Prepare data (no label needed)
        try:
            X = klines_data.copy()
            y_pred_proba = self.model.predict_proba(X)
            latest_proba = y_pred_proba[-1]
            if latest_proba[1] > 0.5:
                direction = "UP"
                confidence = latest_proba[1] * 100
            else:
                direction = "DOWN"
                confidence = latest_proba[0] * 100
            recommendation = (
                "BUY" if direction == "UP" and confidence >= self.confidence_threshold else
                "SELL" if direction == "DOWN" and confidence >= self.confidence_threshold else
                "HOLD"
            )
            current_price = klines_data['close'].iloc[-1]
            return {
                'direction': direction,
                'recommendation': recommendation,
                'confidence': confidence,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_take_profit(self, entry_price: float, atr: float, direction: str = "long", pct: float = 0.04) -> float:
        """
        Calculate take profit price using ATR and/or fixed percentage.

        Args:
            entry_price (float): The entry or reference price.
            atr (float): Average True Range value.
            direction (str): 'long' or 'short'.
            pct (float): Take profit as a decimal (e.g., 0.04 for 4%).

        Returns:
            float: Take profit price.
        """
        if direction == "long":
            tp_atr = entry_price + 1.5 * atr
            tp_pct = entry_price * (1 + pct)
            return round(max(tp_atr, tp_pct), 4)
        else:
            tp_atr = entry_price - 1.5 * atr
            tp_pct = entry_price * (1 - pct)
            return round(min(tp_atr, tp_pct), 4)

    def _generate_limit_order_recommendations(self, klines_data, recommendation, confidence, support_levels, resistance_levels, current_price):
        """
        Generate limit order recommendations based on analysis results

        Args:
            klines_data: DataFrame with price data and indicators
            recommendation: Trading recommendation (BUY, SELL, HOLD)
            confidence: Confidence level of the prediction (0-100)
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            current_price: Current price of the asset

        Returns:
            Dictionary with limit order recommendations
        """
        # Initialize recommendations
        limit_orders = {
            "entry": [],
            "take_profit": [],
            "stop_loss": [],
            "confidence": confidence,
            "timeframe": self.timeframe
        }

        # Get recent price data for volatility calculation
        recent_data = klines_data.tail(20)

        # Calculate average true range (ATR) for volatility-based recommendations
        atr = 0
        if 'atr' in klines_data.columns:
            atr = klines_data['atr'].iloc[-1]
        else:
            # Calculate simple volatility if ATR is not available
            atr = (recent_data['high'] - recent_data['low']).mean()

        # Calculate price volatility as percentage
        volatility_pct = (atr / current_price) * 100

        # Adjust order spacing based on volatility
        # Higher volatility = wider spacing
        spacing_factor = max(0.5, min(2.0, volatility_pct / 2))

        # Get recent high and low
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()

        # Filter support and resistance levels to get the most relevant ones
        relevant_supports = [s for s in support_levels if s < current_price]
        relevant_resistances = [r for r in resistance_levels if r > current_price]

        # Sort levels by proximity to current price
        relevant_supports.sort(reverse=True)  # Closest first
        relevant_resistances.sort()  # Closest first

        # For BUY recommendation
        if recommendation == "BUY":
            confidence_factor = confidence / 100  # 0 to 1
            primary_entry = current_price * (1 - (0.01 * spacing_factor * (1 - confidence_factor)))
            limit_orders["entry"].append({
                "price": round(primary_entry, 4),
                "type": "LIMIT",
                "description": "Primary entry point slightly below current price",
                "confidence": "HIGH"
            })
            for i, support in enumerate(relevant_supports[:2]):
                if support < primary_entry * 0.98:
                    confidence_level = "MEDIUM" if i == 0 else "LOW"
                    limit_orders["entry"].append({
                        "price": round(support, 4),
                        "type": "LIMIT",
                        "description": f"Support level entry point",
                        "confidence": confidence_level
                    })
            # Take profit: always add robust ATR/percentage-based TP first
            tp_price = self._calculate_take_profit(primary_entry, atr, direction="long")
            limit_orders["take_profit"].append({
                "price": tp_price,
                "type": "LIMIT",
                "description": f"Take profit (ATR/percent based)",
                "confidence": "HIGH"
            })
            # Add resistance-based TPs as secondary
            for i, resistance in enumerate(relevant_resistances[:3]):
                confidence_level = "HIGH" if i == 0 else "MEDIUM" if i == 1 else "LOW"
                limit_orders["take_profit"].append({
                    "price": round(resistance, 4),
                    "type": "LIMIT",
                    "description": f"Take profit at resistance level",
                    "confidence": confidence_level
                })
            # Stop loss below nearest support
            if relevant_supports:
                stop_price = relevant_supports[0] * (1 - (0.02 * spacing_factor))
                limit_orders["stop_loss"].append({
                    "price": round(stop_price, 4),
                    "type": "STOP",
                    "description": "Stop loss below nearest support level",
                    "confidence": "HIGH"
                })
            else:
                stop_price = primary_entry - (0.05 * spacing_factor * primary_entry)
                limit_orders["stop_loss"].append({
                    "price": round(stop_price, 4),
                    "type": "STOP",
                    "description": f"Stop loss at {(0.05 * spacing_factor) * 100:.1f}% below entry",
                    "confidence": "MEDIUM"
                })
        # For SELL recommendation
        elif recommendation == "SELL":
            confidence_factor = confidence / 100  # 0 to 1
            primary_entry = current_price * (1 + (0.01 * spacing_factor * (1 - confidence_factor)))
            limit_orders["entry"].append({
                "price": round(primary_entry, 4),
                "type": "LIMIT",
                "description": "Primary entry point slightly above current price",
                "confidence": "HIGH"
            })
            for i, resistance in enumerate(relevant_resistances[:2]):
                if resistance > primary_entry * 1.02:
                    confidence_level = "MEDIUM" if i == 0 else "LOW"
                    limit_orders["entry"].append({
                        "price": round(resistance, 4),
                        "type": "LIMIT",
                        "description": f"Resistance level entry point",
                        "confidence": confidence_level
                    })
            # Take profit: always add robust ATR/percentage-based TP first
            tp_price = self._calculate_take_profit(primary_entry, atr, direction="short")
            limit_orders["take_profit"].append({
                "price": tp_price,
                "type": "LIMIT",
                "description": f"Take profit (ATR/percent based)",
                "confidence": "HIGH"
            })
            # Add support-based TPs as secondary
            for i, support in enumerate(relevant_supports[:3]):
                confidence_level = "HIGH" if i == 0 else "MEDIUM" if i == 1 else "LOW"
                limit_orders["take_profit"].append({
                    "price": round(support, 4),
                    "type": "LIMIT",
                    "description": f"Take profit at support level",
                    "confidence": confidence_level
                })
            # Stop loss above nearest resistance
            if relevant_resistances:
                stop_price = relevant_resistances[0] * (1 + (0.02 * spacing_factor))
                limit_orders["stop_loss"].append({
                    "price": round(stop_price, 4),
                    "type": "STOP",
                    "description": "Stop loss above nearest resistance level",
                    "confidence": "HIGH"
                })
            else:
                stop_price = primary_entry + (0.05 * spacing_factor * primary_entry)
                limit_orders["stop_loss"].append({
                    "price": round(stop_price, 4),
                    "type": "STOP",
                    "description": f"Stop loss at {(0.05 * spacing_factor) * 100:.1f}% above entry",
                    "confidence": "MEDIUM"
                })

        # For HOLD recommendation
        else:
            # For HOLD, provide both BUY and SELL opportunities at key levels

            # BUY opportunities at support levels
            for i, support in enumerate(relevant_supports[:2]):
                confidence_level = "MEDIUM" if i == 0 else "LOW"
                limit_orders["entry"].append({
                    "price": support,
                    "type": "LIMIT_BUY",
                    "description": f"Potential buy at support level",
                    "confidence": confidence_level
                })

            # SELL opportunities at resistance levels
            for i, resistance in enumerate(relevant_resistances[:2]):
                confidence_level = "MEDIUM" if i == 0 else "LOW"
                limit_orders["entry"].append({
                    "price": resistance,
                    "type": "LIMIT_SELL",
                    "description": f"Potential sell at resistance level",
                    "confidence": confidence_level
                })

            # Add breakout levels
            if recent_high > current_price * 1.01:
                limit_orders["entry"].append({
                    "price": recent_high * 1.01,
                    "type": "STOP_BUY",
                    "description": "Breakout buy above recent high",
                    "confidence": "LOW"
                })

            if recent_low < current_price * 0.99:
                limit_orders["entry"].append({
                    "price": recent_low * 0.99,
                    "type": "STOP_SELL",
                    "description": "Breakdown sell below recent low",
                    "confidence": "LOW"
                })

        # Add risk-reward ratios for each entry/stop loss pair
        for entry in limit_orders["entry"]:
            if "BUY" in entry["type"] or entry["type"] == "LIMIT":
                for stop in limit_orders["stop_loss"]:
                    if entry["price"] > stop["price"]:  # Valid for buy orders
                        risk = entry["price"] - stop["price"]
                        for tp in limit_orders["take_profit"]:
                            if tp["price"] > entry["price"]:  # Valid take profit for buy
                                reward = tp["price"] - entry["price"]
                                ratio = reward / risk if risk > 0 else 0
                                if "risk_reward" not in entry:
                                    entry["risk_reward"] = []
                                entry["risk_reward"].append({
                                    "stop_price": stop["price"],
                                    "take_profit_price": tp["price"],
                                    "ratio": ratio,
                                    "quality": "EXCELLENT" if ratio > 3 else "GOOD" if ratio > 2 else "FAIR" if ratio > 1 else "POOR"
                                })
            elif "SELL" in entry["type"]:
                for stop in limit_orders["stop_loss"]:
                    if entry["price"] < stop["price"]:  # Valid for sell orders
                        risk = stop["price"] - entry["price"]
                        for tp in limit_orders["take_profit"]:
                            if tp["price"] < entry["price"]:  # Valid take profit for sell
                                reward = entry["price"] - tp["price"]
                                ratio = reward / risk if risk > 0 else 0
                                if "risk_reward" not in entry:
                                    entry["risk_reward"] = []
                                entry["risk_reward"].append({
                                    "stop_price": stop["price"],
                                    "take_profit_price": tp["price"],
                                    "ratio": ratio,
                                    "quality": "EXCELLENT" if ratio > 3 else "GOOD" if ratio > 2 else "FAIR" if ratio > 1 else "POOR"
                                })

        # Add overall recommendation summary
        limit_orders["summary"] = {
            "best_entry": None,
            "best_take_profit": None,
            "best_stop_loss": None,
            "best_risk_reward_ratio": 0,
            "recommendation": recommendation
        }

        # Find the best risk-reward setup
        best_ratio = 0
        best_entry = None
        best_tp = None
        best_sl = None

        for entry in limit_orders["entry"]:
            if "risk_reward" in entry:
                for rr in entry["risk_reward"]:
                    if rr["ratio"] > best_ratio:
                        best_ratio = rr["ratio"]
                        best_entry = entry["price"]
                        best_tp = rr["take_profit_price"]
                        best_sl = rr["stop_price"]

        if best_ratio > 0:
            limit_orders["summary"]["best_entry"] = best_entry
            limit_orders["summary"]["best_take_profit"] = best_tp
            limit_orders["summary"]["best_stop_loss"] = best_sl
            limit_orders["summary"]["best_risk_reward_ratio"] = best_ratio

        return limit_orders

    def analyze_market_data(self, symbol, klines_data, ticker_data=None, additional_context=None):
        """
        Analyze market data using ML and generate trading recommendations

        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            klines_data: DataFrame containing historical price data
            ticker_data: Current ticker information (optional)
            additional_context: Any additional context to provide (optional)
                - timeframe: The timeframe used for analysis (e.g., '1h', '4h', '1d')

        Returns:
            Dictionary containing analysis results and trading recommendation
        """
        self.ticker_data = ticker_data
        if additional_context and 'timeframe' in additional_context:
            self.timeframe = additional_context['timeframe']
        try:
            if klines_data is None or len(klines_data) < 50:
                return {
                    "error": "Not enough data for ML analysis",
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "recommendation": "HOLD",
                    "confidence": 0,
                    "reasoning": "Not enough historical data for analysis"
                }
            klines_data = klines_data.copy()
            if 'rsi' not in klines_data.columns:
                try:
                    klines_data = calculate_technical_indicators(klines_data)
                except Exception as e:
                    return {
                        "error": f"Error calculating technical indicators: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "recommendation": "HOLD",
                        "confidence": 0,
                        "reasoning": "Error in technical analysis"
                    }
            if len(klines_data.dropna()) < 30:
                return {
                    "error": "Insufficient data after calculating indicators",
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "recommendation": "HOLD",
                    "confidence": 0,
                    "reasoning": "Not enough clean data points for analysis"
                }
            # --- Compute support/resistance ---
            support_levels = compute_support_levels(klines_data, window=20)
            resistance_levels = compute_resistance_levels(klines_data, window=20)
            if self.model is None:
                try:
                    training_results = self.train_model(klines_data)
                    if 'error' in training_results:
                        return {
                            "error": f"Error training model: {training_results['error']}",
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "recommendation": "HOLD",
                            "confidence": 0,
                            "reasoning": "Could not train ML model"
                        }
                except Exception as e:
                    return {
                        "error": f"Error training model: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "recommendation": "HOLD",
                        "confidence": 0,
                        "reasoning": "Error in model training"
                    }
            try:
                prediction = self.predict(klines_data)
                if 'error' in prediction:
                    return {
                        "error": prediction['error'],
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "recommendation": "HOLD",
                        "confidence": 0,
                        "reasoning": prediction.get('reasoning', 'Error in prediction')
                    }
            except Exception as e:
                return {
                    "error": f"Error making prediction: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "recommendation": "HOLD",
                    "confidence": 0,
                    "reasoning": "Error in prediction process"
                }
            prediction['symbol'] = symbol
            if prediction.get('recommendation') == "BUY":
                prediction['market_sentiment'] = "BULLISH"
            elif prediction.get('recommendation') == "SELL":
                prediction['market_sentiment'] = "BEARISH"
            else:
                prediction['market_sentiment'] = "NEUTRAL"
            prediction['position_size_recommendation'] = min(prediction.get('confidence', 0), 100)
            confidence = prediction.get('confidence', 0)
            prediction['timeframe_suitability'] = "HIGH" if confidence > 80 else "MEDIUM" if confidence > 60 else "LOW"
            # --- Always pass support/resistance to limit order generator ---
            limit_orders = self._generate_limit_order_recommendations(
                klines_data,
                prediction.get('recommendation', 'HOLD'),
                prediction.get('confidence', 0),
                support_levels,
                resistance_levels,
                prediction.get('current_price', 0)
            )
            prediction['limit_orders'] = limit_orders
            prediction['support_levels'] = support_levels
            prediction['resistance_levels'] = resistance_levels
            # --- Always set stop_loss and take_profit at top level ---
            prediction['stop_loss'] = limit_orders['stop_loss'][0]['price'] if limit_orders['stop_loss'] else None
            prediction['take_profit'] = limit_orders['take_profit'][0]['price'] if limit_orders['take_profit'] else None
            # --- Always set key_indicators ---
            latest = klines_data.iloc[-1]
            key_indicators = {}
            for col in ['rsi', 'macd', 'macd_signal', 'ema_9', 'ema_20', 'ema_50', 'ema_100', 'bbands_u', 'bbands_l', 'atr_14', 'adx_14', 'obv', 'willr_14']:
                if col in latest:
                    key_indicators[col] = float(latest[col])
            prediction['key_indicators'] = key_indicators if key_indicators else None
            # --- Always set top_features ---
            top_features = None
            if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                model_step = self.model.named_steps['model']
                if hasattr(model_step, 'feature_importances_'):
                    X_df, feature_names = self._prepare_features(klines_data)
                    importances = model_step.feature_importances_
                    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    top_features = [f for f, imp in feat_imp[:5]]
            prediction['top_features'] = top_features
            # --- Tangible, Actionable Output ---
            output = {
                "symbol": symbol,
                "timeframe": self.timeframe,
                "recommendation": prediction.get('recommendation'),
                "direction": prediction.get('direction'),
                "confidence": prediction.get('confidence'),
                "current_price": prediction.get('current_price'),
                "support_levels": prediction.get('support_levels'),
                "resistance_levels": prediction.get('resistance_levels'),
                "stop_loss": prediction.get('stop_loss'),
                "take_profit": prediction.get('take_profit'),
                "key_indicators": prediction.get('key_indicators'),
                "top_features": prediction.get('top_features'),
                "limit_orders": prediction.get('limit_orders'),
                "metrics": {
                    "market_sentiment": prediction.get('market_sentiment'),
                    "position_size_recommendation": prediction.get('position_size_recommendation'),
                    "timeframe_suitability": prediction.get('timeframe_suitability'),
                },
                "entry_points": [entry['price'] for entry in prediction.get('limit_orders', {}).get('entry', [])],
                "exit_points": [tp['price'] for tp in prediction.get('limit_orders', {}).get('take_profit', [])],
                "stop_loss_points": [sl['price'] for sl in prediction.get('limit_orders', {}).get('stop_loss', [])],
            }

            # --- Brief Price Action Explanation ---
            pa_summary = []
            pa_summary.append(f"Latest close: {latest['close']:.2f}, open: {latest['open']:.2f}, high: {latest['high']:.2f}, low: {latest['low']:.2f}, volume: {latest['volume']:.2f}.")
            if 'rsi' in latest:
                if latest['rsi'] > 70:
                    pa_summary.append(f"RSI is overbought ({latest['rsi']:.1f}). Potential for pullback.")
                elif latest['rsi'] < 30:
                    pa_summary.append(f"RSI is oversold ({latest['rsi']:.1f}). Potential for bounce.")
                else:
                    pa_summary.append(f"RSI is neutral ({latest['rsi']:.1f}).")
            if 'macd' in latest and 'macd_signal' in latest:
                if latest['macd'] > latest['macd_signal']:
                    pa_summary.append("MACD is above signal line (bullish momentum).")
                else:
                    pa_summary.append("MACD is below signal line (bearish momentum).")
            if 'bb_high' in latest and 'bb_low' in latest:
                if latest['close'] > latest['bb_high']:
                    pa_summary.append("Price closed above upper Bollinger Band (overextension).")
                elif latest['close'] < latest['bb_low']:
                    pa_summary.append("Price closed below lower Bollinger Band (potential reversal).")
            if 'adx' in latest:
                if latest['adx'] > 25:
                    pa_summary.append(f"Strong trend (ADX: {latest['adx']:.1f}).")
                elif latest['adx'] < 20:
                    pa_summary.append(f"Weak trend (ADX: {latest['adx']:.1f}).")
            output["price_action_summary"] = " ".join(pa_summary)
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 0,
                "reasoning": "Error in analysis"
            }

    def backtest(
        self,
        klines_data: pd.DataFrame,
        n_splits: int = 5,
        initial_balance: float = 10000.0,
        fee: float = 0.001,
        strategy: Optional[Callable[[pd.DataFrame], np.ndarray]] = None
    ) -> dict:
        """
        Perform walk-forward backtesting with the current ML pipeline or a user-selected strategy.

        Args:
            klines_data (pd.DataFrame): Historical price data
            n_splits (int): Number of walk-forward splits
            initial_balance (float): Starting balance for trading simulation
            fee (float): Trading fee per trade (fraction)
            strategy (Optional[Callable]): If provided, use this strategy function for signal generation.

        Returns:
            dict: Backtest results with mean and per-split metrics, trading simulation
        """
        results = {
            'algorithm': self.algorithm if strategy is None else getattr(strategy, '__name__', str(strategy)),
            'splits': n_splits,
            'split_metrics': [],
            'timestamp': datetime.now().isoformat()
        }
        klines_data = calculate_technical_indicators(klines_data)
        y, labeled_df = self._prepare_labels(klines_data, horizon=self.prediction_horizon)
        y_series = pd.Series(y, index=labeled_df.index)
        mask = ~klines_data.isnull().any(axis=1) & ~y_series.isnull()
        X_clean = klines_data[mask]
        y_clean = y_series[mask]
        close_prices = klines_data.loc[mask, 'close'].values
        if len(X_clean) < n_splits * 10:
            results['error'] = 'Not enough data for backtest'
            return results
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs, f1s, precisions, recalls, aucs, bal_accs = [], [], [], [], [], []
        split_conf_matrices = []
        split_returns = []
        split_drawdowns = []
        split_sharpes = []
        equity_curve_full = []
        timestamps_full = []
        # If klines_data has a timestamp column, use it
        if 'timestamp' in klines_data.columns:
            all_timestamps = klines_data[mask]['timestamp'].values
        else:
            all_timestamps = np.arange(len(X_clean))
        for split_idx, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            prices_test = close_prices[test_idx]
            split_timestamps = all_timestamps[test_idx]
            if strategy is not None:
                y_pred = strategy(X_test)
                y_pred = np.where(y_pred == -1, 0, y_pred)
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', self._create_model())
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_proba = self.model.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            cm = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            # Trading simulation
            balance = initial_balance
            position = 0
            equity_curve = [balance]
            for i, (pred, true, price) in enumerate(zip(y_pred, y_test, prices_test)):
                if pred == 1 and position == 0:
                    position = 1
                    entry_price = price * (1 + fee)
                elif pred == 0 and position == 0:
                    position = -1
                    entry_price = price * (1 - fee)
                elif position != 0:
                    pnl = (price - entry_price) if position == 1 else (entry_price - price)
                    balance += pnl
                    balance -= abs(pnl) * fee
                    position = 0
                equity_curve.append(balance)
            # For equity curve output, align with timestamps
            # The first equity is before the first test bar, so skip it for plotting
            equity_curve_split = equity_curve[1:]
            equity_curve_full.extend(equity_curve_split)
            timestamps_full.extend(split_timestamps)
            returns = np.diff(equity_curve)
            cum_return = (equity_curve[-1] - initial_balance) / initial_balance
            max_drawdown = 0
            peak = equity_curve[0]
            for x in equity_curve:
                if x > peak:
                    peak = x
                dd = (peak - x) / peak
                if dd > max_drawdown:
                    max_drawdown = dd
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            accs.append(acc)
            f1s.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            bal_accs.append(bal_acc)
            if auc is not None:
                aucs.append(auc)
            split_conf_matrices.append(cm)
            split_returns.append(cum_return)
            split_drawdowns.append(max_drawdown)
            split_sharpes.append(sharpe)
            results['split_metrics'].append({
                'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec,
                'balanced_accuracy': bal_acc, 'auc': auc, 'confusion_matrix': cm,
                'cumulative_return': cum_return, 'max_drawdown': max_drawdown, 'sharpe': sharpe,
                'classification_report': class_report
            })
        results['accuracy'] = float(np.mean(accs)) if accs else 0.0
        results['f1'] = float(np.mean(f1s)) if f1s else 0.0
        results['precision'] = float(np.mean(precisions)) if precisions else 0.0
        results['recall'] = float(np.mean(recalls)) if recalls else 0.0
        results['balanced_accuracy'] = float(np.mean(bal_accs)) if bal_accs else 0.0
        results['auc'] = float(np.mean(aucs)) if aucs else None
        results['confusion_matrices'] = split_conf_matrices
        results['mean_cumulative_return'] = float(np.mean(split_returns)) if split_returns else 0.0
        results['mean_max_drawdown'] = float(np.mean(split_drawdowns)) if split_drawdowns else 0.0
        results['mean_sharpe'] = float(np.mean(split_sharpes)) if split_sharpes else 0.0
        # Add equity curve for plotting/saving
        results['equity_curve'] = [
            {'timestamp': int(ts), 'equity': float(eq)}
            for ts, eq in zip(timestamps_full, equity_curve_full)
        ]
        return results

    def backtest_limit_orders(self, klines_data: pd.DataFrame, fee: float = 0.001) -> dict:
        """
        Backtest the limit order recommendations over historical data.

        Args:
            klines_data (pd.DataFrame): Historical OHLCV data (must include open, high, low, close).
            fee (float): Trading fee per trade (fraction).

        Returns:
            dict: Backtest results and trade log (win rate, avg risk/reward, total trades, total pnl, trades list).
        """
        trades = []
        klines_data = klines_data.reset_index(drop=True)
        for i in range(len(klines_data) - 1):
            df_slice = klines_data.iloc[:i+1]
            row = klines_data.iloc[i]
            next_row = klines_data.iloc[i + 1]
            # Generate limit orders for this bar
            rec = self.predict(df_slice)
            support = compute_support_levels(df_slice, window=20)
            resistance = compute_resistance_levels(df_slice, window=20)
            orders = self._generate_limit_order_recommendations(
                df_slice,
                rec.get("recommendation", "HOLD"),
                rec.get("confidence", 0),
                support,
                resistance,
                rec.get("current_price", row["close"])
            )
            for entry in orders.get("entry", []):
                entry_price = entry["price"]
                # Simulate if entry is hit in next bar
                if next_row["low"] <= entry_price <= next_row["high"]:
                    # Find corresponding TP/SL
                    for tp in orders.get("take_profit", []):
                        tp_price = tp["price"]
                        hit_tp = (next_row["high"] >= tp_price) if tp_price > entry_price else (next_row["low"] <= tp_price)
                        for sl in orders.get("stop_loss", []):
                            sl_price = sl["price"]
                            hit_sl = (next_row["low"] <= sl_price) if sl_price < entry_price else (next_row["high"] >= sl_price)
                            # Determine which is hit first (TP wins if both hit)
                            if hit_tp and (not hit_sl or next_row["high"] >= tp_price):
                                pnl = (tp_price - entry_price) - fee * entry_price
                                trades.append({"entry": entry_price, "exit": tp_price, "type": "TP", "pnl": pnl, "i": i})
                            elif hit_sl:
                                pnl = (sl_price - entry_price) - fee * entry_price
                                trades.append({"entry": entry_price, "exit": sl_price, "type": "SL", "pnl": pnl, "i": i})
        wins = [t for t in trades if t["type"] == "TP"]
        losses = [t for t in trades if t["type"] == "SL"]
        win_rate = len(wins) / len(trades) if trades else 0
        avg_risk_reward = (
            sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses else 0
        )
        return {
            "trades": trades,
            "win_rate": win_rate,
            "avg_risk_reward": avg_risk_reward,
            "total_trades": len(trades),
            "total_pnl": sum(t["pnl"] for t in trades)
        }

    def optimize_with_optuna(
        self,
        klines_data: pd.DataFrame,
        model_type: str = "RandomForest",
        n_trials: int = 20,
        direction: str = "maximize",
        study_name: str = "ml_optuna_study",
        storage: Optional[str] = None,
        show_progress_bar: bool = True,
    ) -> Study:
        """
        Optimize model hyperparameters using Optuna and track multiple metrics for each trial.

        Args:
            klines_data (pd.DataFrame): Historical price data for training/validation.
            model_type (str): Model to optimize ("RandomForest", "XGBoost", "SVM").
            n_trials (int): Number of Optuna trials.
            direction (str): "maximize" or "minimize" the objective (accuracy).
            study_name (str): Name for the Optuna study.
            storage (Optional[str]): Optuna storage URI for persistent studies.
            show_progress_bar (bool): Whether to show Optuna's progress bar.

        Returns:
            Study: The Optuna study object with all trial results.
        """
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        def objective(trial: Trial) -> float:
            # Model selection and hyperparameter search space
            if model_type == "RandomForest":
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
                model = sklearn.ensemble.RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "XGBoost":
                from xgboost import XGBClassifier
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1
                )
            elif model_type == "SVM":
                c = trial.suggest_float("C", 0.1, 10.0, log=True)
                gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
                model = sklearn.svm.SVC(
                    kernel='rbf',
                    C=c,
                    gamma=gamma,
                    probability=True,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Prepare data
            data = calculate_technical_indicators(klines_data)
            y, labeled_df = self._prepare_labels(data, horizon=self.prediction_horizon)
            y_series = pd.Series(y, index=labeled_df.index)
            mask = ~data.isnull().any(axis=1) & ~y_series.isnull()
            X_clean = data[mask]
            y_clean = y_series[mask]
            if len(X_clean) < 50:
                trial.report(0.0, step=0)
                return 0.0
            X_train, X_valid, y_train, y_valid = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            y_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred, zero_division=0)
            prec = precision_score(y_valid, y_pred, zero_division=0)
            rec = recall_score(y_valid, y_pred, zero_division=0)
            auc = roc_auc_score(y_valid, y_proba) if y_proba is not None else None

            # Log all metrics
            trial.set_user_attr("accuracy", acc)
            trial.set_user_attr("f1", f1)
            trial.set_user_attr("precision", prec)
            trial.set_user_attr("recall", rec)
            trial.set_user_attr("auc", auc)

            return acc  # or f1, depending on your optimization goal

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
        return study

    def parameter_sweep_backtest(
        self,
        klines_data: pd.DataFrame,
        param_grid: dict,
        n_splits: int = 5,
        initial_balance: float = 10000.0,
        fee: float = 0.001,
        strategy: Optional[Callable[[pd.DataFrame], np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Run backtest over a grid of parameters and return a DataFrame of results.

        Args:
            klines_data (pd.DataFrame): Historical price data.
            param_grid (dict): Dictionary of parameter lists, e.g. {'fee': [0.001, 0.002], 'n_splits': [3, 5]}.
            n_splits (int): Default number of splits.
            initial_balance (float): Default starting balance.
            fee (float): Default trading fee.
            strategy (Optional[Callable]): Optional strategy function.

        Returns:
            pd.DataFrame: DataFrame with one row per parameter combination and backtest results.
        """
        # Prepare all combinations of parameters
        keys, values = zip(*param_grid.items())
        param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        results = []
        for params in param_combos:
            # Use provided params or defaults
            ns = params.get('n_splits', n_splits)
            bal = params.get('initial_balance', initial_balance)
            f = params.get('fee', fee)
            # Optionally, pass params to strategy if it supports them
            try:
                res = self.backtest(
                    klines_data=klines_data,
                    n_splits=ns,
                    initial_balance=bal,
                    fee=f,
                    strategy=strategy
                )
                # Flatten results for DataFrame
                flat = {**params}
                flat.update({
                    'accuracy': res.get('accuracy'),
                    'f1': res.get('f1'),
                    'precision': res.get('precision'),
                    'recall': res.get('recall'),
                    'mean_cumulative_return': res.get('mean_cumulative_return'),
                    'mean_max_drawdown': res.get('mean_max_drawdown'),
                    'mean_sharpe': res.get('mean_sharpe'),
                    'error': res.get('error', None)
                })
                results.append(flat)
            except Exception as e:
                flat = {**params, 'error': str(e)}
                results.append(flat)
        return pd.DataFrame(results)

def format_ml_training_result(result: dict) -> str:
    lines = []
    lines.append(f"ML Model Training Summary")
    lines.append(f"Algorithm: {result.get('algorithm')}")
    lines.append(f"Accuracy: {result.get('accuracy'):.2f}")
    lines.append(f"Precision: {result.get('precision'):.2f}")
    lines.append(f"Recall: {result.get('recall'):.2f}")
    lines.append(f"F1 Score: {result.get('f1_score'):.2f}")
    lines.append(f"Training Time: {result.get('training_time'):.2f} seconds")
    lines.append(f"Timestamp: {result.get('timestamp')}")
    lines.append("")
    # Confusion Matrix
    cm = result.get('confusion_matrix')
    if cm:
        lines.append("Confusion Matrix:")
        lines.append("            Pred 0   Pred 1")
        lines.append(f"Actual 0     {cm[0][0]:>5}     {cm[0][1]:>5}")
        lines.append(f"Actual 1     {cm[1][0]:>5}     {cm[1][1]:>5}")
        lines.append("")
    # Feature Importances
    fi = result.get('feature_importances', {})
    if fi:
        lines.append("Top Feature Importances:")
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_fi[:10]:
            lines.append(f"  {feat:15s} {imp:.4f}")
    return "\n".join(lines)

def plot_optuna_results(study: optuna.Study) -> None:
    """
    Visualize Optuna study results: optimization history, parameter importance, and parallel coordinates.

    Args:
        study (optuna.Study): The Optuna study object to visualize.

    Returns:
        None

    Example:
        >>> study = analyzer.optimize_with_optuna(df, model_type="XGBoost", n_trials=50)
        >>> plot_optuna_results(study)
    """
    import optuna.visualization
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()

def launch_optuna_dashboard(storage: str, port: int = 8080) -> None:
    """
    Launch the Optuna dashboard for interactive study visualization.

    Args:
        storage (str): The Optuna storage URI (e.g., 'sqlite:///example.db').
        port (int): The port to run the dashboard on (default: 8080).

    Returns:
        None

    Example:
        >>> launch_optuna_dashboard('sqlite:///example.db', port=8080)
    """
    try:
        print(f"Launching Optuna dashboard at http://localhost:{port} ...")
        subprocess.Popen([
            "optuna", "dashboard", storage, "--port", str(port)
        ])
    except Exception as e:
        print(f"Failed to launch Optuna dashboard: {e}")
