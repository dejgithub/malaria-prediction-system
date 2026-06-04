import numpy as np
from datetime import datetime


class InsightsEngine:
    def __init__(self, forecasts, historical_data=None):
        self.forecasts = forecasts
        self.historical_data = historical_data

    def generate_insights(self):
        insights = []
        if not self.forecasts:
            return insights

        forecasts_data = self.forecasts.get("forecasts", {})
        overall_risk = self.forecasts.get("overall_risk_assessment", "Unknown")
        peak = self.forecasts.get("peak_prediction_period", "Unknown")

        insights.append({
            "type": "risk",
            "severity": "high" if "Critical" in overall_risk else "medium" if "High" in overall_risk else "low",
            "message": f"Overall malaria risk assessment: {overall_risk}",
            "detail": "Based on hybrid deep learning model predictions across all forecast periods."
        })

        trends = []
        for period, data in forecasts_data.items():
            trend = data.get("trend_direction", "Stable")
            if trend != "Stable":
                trends.append(f"{period.replace('_', ' ')}: {trend}")

        if trends:
            insights.append({
                "type": "trend",
                "severity": "medium",
                "message": "Trend analysis: " + "; ".join(trends),
                "detail": "Model detects significant directional changes in forecast periods."
            })
        else:
            insights.append({
                "type": "trend",
                "severity": "low",
                "message": "Trends are stable across all forecast periods.",
                "detail": "No significant directional changes detected."
            })

        periods_with_risk = []
        for period, data in forecasts_data.items():
            risk = data.get("risk_level", "Low Risk")
            if risk in ["High Risk", "Critical Risk"]:
                periods_with_risk.append(f"{period.replace('_', ' ')} ({risk})")

        if periods_with_risk:
            insights.append({
                "type": "warning",
                "severity": "high",
                "message": f"High risk periods detected: {', '.join(periods_with_risk)}",
                "detail": "Immediate attention and resource allocation recommended for these periods."
            })

        insights.append({
            "type": "peak",
            "severity": "medium",
            "message": f"Predicted peak period: {peak}",
            "detail": "Plan resource allocation and intervention strategies around this period."
        })

        if self.historical_data is not None and len(self.historical_data) > 0:
            recent_trend = self._analyze_recent_trend()
            if recent_trend:
                insights.append(recent_trend)

        seasonal = self._detect_seasonal_patterns()
        if seasonal:
            insights.append(seasonal)

        return insights

    def _analyze_recent_trend(self):
        if len(self.historical_data) < 6:
            return None
        recent = self.historical_data[-6:]
        mid = len(recent) // 2
        first_half = np.mean(recent[:mid])
        second_half = np.mean(recent[mid:])
        change_pct = ((second_half - first_half) / (first_half + 1)) * 100

        if change_pct > 10:
            return {
                "type": "recent_trend",
                "severity": "high" if change_pct > 25 else "medium",
                "message": f"Recent cases {['decreased', 'increased'][change_pct > 0]} by {abs(change_pct):.1f}%",
                "detail": f"Based on the last 6 months of historical data analysis."
            }
        return None

    def _detect_seasonal_patterns(self):
        if self.historical_data is None or len(self.historical_data) < 12:
            return None
        monthly_avgs = []
        for i in range(12):
            month_data = self.historical_data[i::12]
            if len(month_data) > 0:
                monthly_avgs.append(np.mean(month_data))
        if monthly_avgs:
            peak_month = np.argmax(monthly_avgs) + 1
            low_month = np.argmin(monthly_avgs) + 1
            return {
                "type": "seasonal",
                "severity": "medium",
                "message": f"Seasonal pattern detected: Peak around month {peak_month}, low around month {low_month}",
                "detail": "Historical data shows recurring seasonal variations in malaria cases."
            }
        return None

    def generate_recommendations(self):
        recommendations = []
        if not self.forecasts:
            return self._default_recommendations()

        overall_risk = self.forecasts.get("overall_risk_assessment", "Low Risk")
        forecasts_data = self.forecasts.get("forecasts", {})

        if "Critical" in overall_risk:
            recommendations.append({
                "priority": "immediate",
                "action": "Deploy emergency malaria response teams",
                "rationale": "Critical risk level detected across forecast periods",
                "timeline": "Within 1 week"
            })
            recommendations.append({
                "priority": "immediate",
                "action": "Distribute insecticide-treated nets and antimalarial drugs",
                "rationale": "Urgent need to reduce transmission and provide treatment",
                "timeline": "Within 2 weeks"
            })

        if "High" in overall_risk or "Critical" in overall_risk:
            recommendations.append({
                "priority": "high",
                "action": "Increase healthcare facility preparedness",
                "rationale": "Expected surge in malaria cases requires additional resources",
                "timeline": "Within 1 month"
            })
            recommendations.append({
                "priority": "high",
                "action": "Launch community awareness and prevention campaigns",
                "rationale": "Proactive community engagement reduces transmission risk",
                "timeline": "Ongoing"
            })

        recommendations.append({
            "priority": "medium",
            "action": "Enhance surveillance and monitoring systems",
            "rationale": "Early detection enables timely intervention",
            "timeline": "Ongoing"
        })

        recommendations.append({
            "priority": "medium",
            "action": "Implement indoor residual spraying before rainy season",
            "rationale": "Vector control is most effective before peak transmission periods",
            "timeline": "Before next rainy season"
        })

        recommendations.append({
            "priority": "low",
            "action": "Strengthen health information systems",
            "rationale": "Accurate real-time data improves prediction accuracy",
            "timeline": "Long-term"
        })

        recommendations.append({
            "priority": "low",
            "action": "Conduct training programs for healthcare workers",
            "rationale": "Well-trained staff improves diagnosis and treatment outcomes",
            "timeline": "Quarterly"
        })

        return recommendations

    def _default_recommendations(self):
        return [
            {
                "priority": "medium",
                "action": "Maintain current malaria prevention strategies",
                "rationale": "Continue established prevention and control measures",
                "timeline": "Ongoing"
            },
            {
                "priority": "medium",
                "action": "Enhance data collection for better predictions",
                "rationale": "More comprehensive data improves model accuracy",
                "timeline": "Ongoing"
            }
        ]
