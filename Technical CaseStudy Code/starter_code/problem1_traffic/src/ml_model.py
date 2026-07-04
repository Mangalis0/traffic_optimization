"""
ML traffic predictor.

Uses GradientBoostingRegressor to predict per-approach arrival rates from
time-of-day, day-of-week, weather, and location features.

Why GBR?
- Handles non-linear rush-hour peaks naturally
- Fast to train on synthetic data (< 5 s for 30 days)
- Good out-of-box performance without extensive tuning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict


FEATURE_COLS = [
    'sin_hour', 'cos_hour',   # cyclic time encoding
    'sin_dow', 'cos_dow',     # cyclic day encoding
    'is_weekend',
    'weather',
    'intersection_id',
    'direction_encoded',
]
TARGET_COL = 'arrival_rate'

_DIRECTIONS = ['N', 'S', 'E', 'W']
_DIR_ENC = {'N': 0, 'S': 1, 'E': 2, 'W': 3}


class TrafficPredictor:
    """
    Predicts arrival rates (veh/hr) for every approach at every intersection.
    """

    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('gbr', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )),
        ])
        self.trained = False

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Fit on historical data and return cross-validation metrics.
        """
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)

        self.model.fit(X, y)
        self.trained = True

        importances = self.model.named_steps['gbr'].feature_importances_
        return {
            'cv_rmse_mean': float(cv_rmse.mean()),
            'cv_rmse_std': float(cv_rmse.std()),
            'feature_importances': dict(zip(FEATURE_COLS, importances.tolist())),
            'train_samples': len(df),
        }

    def predict_rates(
        self,
        hour: float,
        day_of_week: int,
        weather: float,
    ) -> Dict[int, Dict[str, float]]:
        """
        Predict arrival rates for all 16 approach lanes (4 intersections × 4 directions).

        Returns:
            {intersection_id: {direction: rate_vph}}
        """
        assert self.trained, "Call train() first."

        rows = []
        for inter_id in range(4):
            for direction, dir_enc in _DIR_ENC.items():
                rows.append({
                    'sin_hour': np.sin(2 * np.pi * hour / 24),
                    'cos_hour': np.cos(2 * np.pi * hour / 24),
                    'sin_dow': np.sin(2 * np.pi * day_of_week / 7),
                    'cos_dow': np.cos(2 * np.pi * day_of_week / 7),
                    'is_weekend': int(day_of_week >= 5),
                    'weather': weather,
                    'intersection_id': inter_id,
                    'direction_encoded': dir_enc,
                })

        df = pd.DataFrame(rows)
        preds = self.model.predict(df[FEATURE_COLS].values)
        preds = np.maximum(preds, 20.0)

        result: Dict[int, Dict[str, float]] = {}
        idx = 0
        for inter_id in range(4):
            result[inter_id] = {}
            for direction in _DIRECTIONS:
                result[inter_id][direction] = float(preds[idx])
                idx += 1

        return result
