import pandas as pd
import numpy as np
from src.analysis.ml import MLAnalyzer
from config import ML_CONFIG

def test_ml_analyzer_fit_predict():
    # Create dummy data
    df = pd.DataFrame({
        'open': np.arange(100),
        'high': np.arange(100) + 1,
        'low': np.arange(100) - 1,
        'close': np.arange(100),
        'volume': np.random.rand(100) * 100
    })
    df['target'] = np.random.randint(0, 2, size=100)
    analyzer = MLAnalyzer(ML_CONFIG)
    analyzer.fit(df.drop('target', axis=1), df['target'])
    preds = analyzer.predict(df.drop('target', axis=1))
    assert len(preds) == len(df)

def test_ml_analyzer_backtest():
    df = pd.DataFrame({
        'open': np.arange(100),
        'high': np.arange(100) + 1,
        'low': np.arange(100) - 1,
        'close': np.arange(100),
        'volume': np.random.rand(100) * 100,
        'target': np.random.randint(0, 2, size=100)
    })
    analyzer = MLAnalyzer(ML_CONFIG)
    results = analyzer.backtest(df.drop('target', axis=1), df['target'], n_splits=3)
    assert 'accuracy' in results
    assert 'f1' in results 