import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta

from config import (
    SEQUENCE_LENGTH, TARGET_COLUMN, RISK_THRESHOLDS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ForecastEngine:
    def __init__(self, model, scaler, feature_columns):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)

    def generate_forecast(self, last_sequence, steps, historical_max=None):
        if historical_max is None:
            historical_max = 1.0

        current_sequence = last_sequence.copy()
        predictions = []
        confidence_scores = []

        for i in range(steps):
            pred_scaled = self.model.predict(current_sequence[np.newaxis, :, :], verbose=0)
            pred_value = pred_scaled[0, 0]

            dummy = np.zeros((1, self.n_features))
            dummy[0, 0] = pred_value
            pred_actual = self.scaler.inverse_transform(dummy)[0, 0]

            next_step = current_sequence[-1].copy()
            next_step[0] = pred_value
            current_sequence = np.vstack([current_sequence[1:], next_step])

            predictions.append(float(max(0, pred_actual)))

            confidence = max(0, min(1, 1.0 - (i * 0.02)))
            confidence_scores.append(confidence)

        return predictions, confidence_scores

    def classify_risk(self, value, historical_max):
        ratio = value / historical_max if historical_max > 0 else 0
        if ratio <= RISK_THRESHOLDS["Low"]:
            return "Low Risk"
        elif ratio <= RISK_THRESHOLDS["Moderate"]:
            return "Moderate Risk"
        elif ratio <= RISK_THRESHOLDS["High"]:
            return "High Risk"
        else:
            return "Critical Risk"

    def get_trend_direction(self, predictions):
        if len(predictions) < 2:
            return "Stable"
        first_half = np.mean(predictions[: len(predictions) // 2])
        second_half = np.mean(predictions[len(predictions) // 2 :])
        diff = second_half - first_half
        threshold = np.mean(predictions) * 0.05

        if diff > threshold:
            return "Increasing"
        elif diff < -threshold:
            return "Decreasing"
        else:
            return "Stable"

    def generate_forecast_report(self, last_sequence, historical_data):
        if historical_data is None or len(historical_data) == 0:
            historical_max = 1.0
        else:
            historical_max = float(np.max(historical_data))

        forecasts = {}

        for period_name, steps in [("1_month", 1), ("3_month", 3), ("6_month", 6), ("12_month", 12)]:
            preds, confs = self.generate_forecast(last_sequence, steps, historical_max)
            risk_level = self.classify_risk(np.mean(preds), historical_max)
            trend = self.get_trend_direction(preds)

            forecasts[period_name] = {
                "predictions": preds,
                "average": float(np.mean(preds)),
                "max": float(np.max(preds)),
                "min": float(np.min(preds)),
                "risk_level": risk_level,
                "confidence_scores": confs,
                "average_confidence": float(np.mean(confs)),
                "trend_direction": trend,
                "steps": steps
            }

        overall_risk = self._determine_overall_risk(forecasts)
        peak_month = self._find_peak_month(forecasts)

        return {
            "forecasts": forecasts,
            "overall_risk_assessment": overall_risk,
            "peak_prediction_period": peak_month,
            "historical_max": historical_max,
            "generated_at": datetime.now().isoformat()
        }

    def _determine_overall_risk(self, forecasts):
        risk_scores = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2, "Critical Risk": 3}
        total = 0
        count = 0
        for period, data in forecasts.items():
            if data["risk_level"] in risk_scores:
                total += risk_scores[data["risk_level"]]
                count += 1
        avg_score = total / count if count > 0 else 0
        if avg_score <= 0.5:
            return "Low Risk"
        elif avg_score <= 1.5:
            return "Moderate Risk"
        elif avg_score <= 2.5:
            return "High Risk"
        else:
            return "Critical Risk"

    def _find_peak_month(self, forecasts):
        all_preds = []
        for period, data in forecasts.items():
            all_preds.extend(data["predictions"])
        if not all_preds:
            return "Unknown"
        max_idx = int(np.argmax(all_preds))
        return f"Month {max_idx + 1} (Peak value: {all_preds[max_idx]:.1f})"
