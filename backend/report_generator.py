import json
import base64
from pathlib import Path
from datetime import datetime

from config import MODEL_DIR, REPORTS_DIR


class ReportGenerator:
    def __init__(self):
        self.template = ""

    def _load_image_as_base64(self, image_path):
        path = Path(image_path)
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""

    def _generate_charts_html(self, metrics, forecasts):
        if not metrics:
            return ""
        charts_html = """
        <div class="section">
            <h2>Performance Charts</h2>
            <div class="chart-grid">
        """

        for chart_name in ["confusion_matrix", "loss_curves", "accuracy_curves", "actual_vs_predicted"]:
            img_path = MODEL_DIR / f"{chart_name}.png"
            img_b64 = self._load_image_as_base64(img_path)
            if img_b64:
                charts_html += f"""
                <div class="chart-card">
                    <h3>{chart_name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{img_b64}" alt="{chart_name}" style="width:100%;max-width:600px;">
                </div>
                """

        charts_html += "</div></div>"
        return charts_html

    def _generate_forecast_table(self, forecasts):
        if not forecasts:
            return ""

        html = """
        <div class="section">
            <h2>Forecast Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Period</th>
                        <th>Average Cases</th>
                        <th>Max Cases</th>
                        <th>Min Cases</th>
                        <th>Risk Level</th>
                        <th>Trend</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
        """

        for period, data in forecasts.items():
            display_name = period.replace("_", " ").title()
            html += f"""
                    <tr>
                        <td>{display_name}</td>
                        <td>{data.get('average', 0):.1f}</td>
                        <td>{data.get('max', 0):.1f}</td>
                        <td>{data.get('min', 0):.1f}</td>
                        <td><span class="risk-{data.get('risk_level', 'Low').split()[0].lower()}">{data.get('risk_level', 'Unknown')}</span></td>
                        <td>{data.get('trend_direction', 'Unknown')}</td>
                        <td>{(data.get('average_confidence', 0) * 100):.1f}%</td>
                    </tr>
            """

        html += "</tbody></table></div>"
        return html

    def _generate_metrics_table(self, metrics):
        if not metrics:
            return ""

        rows = ""
        metric_keys = {
            "mae": "MAE",
            "mse": "MSE",
            "rmse": "RMSE",
            "r2_score": "R² Score",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score"
        }

        for key, label in metric_keys.items():
            if key in metrics:
                value = metrics[key]
                formatted = f"{value * 100:.2f}%" if key in ["accuracy", "precision", "recall", "f1_score"] else f"{value:.4f}"
                rows += f"<tr><td>{label}</td><td>{formatted}</td></tr>\n"

        return f"""
        <div class="section">
            <h2>Model Performance Metrics</h2>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _generate_recommendations(self, forecasts, metrics):
        recommendations = []

        if forecasts:
            overall_risk = forecasts.get("overall_risk_assessment", "Low Risk")
            if "Critical" in overall_risk:
                recommendations.extend([
                    "URGENT: Immediate intervention required due to critical risk levels.",
                    "Deploy emergency response teams to high-risk regions.",
                    "Distribute mosquito nets and antimalarial medications immediately."
                ])
            elif "High" in overall_risk:
                recommendations.extend([
                    "Increase surveillance and monitoring in affected areas.",
                    "Prepare healthcare facilities for potential surge in cases.",
                    "Launch public awareness campaigns about prevention measures."
                ])
            elif "Moderate" in overall_risk:
                recommendations.extend([
                    "Continue regular monitoring and prevention programs.",
                    "Maintain adequate supply of antimalarial drugs.",
                    "Strengthen community-based malaria prevention initiatives."
                ])
            else:
                recommendations.extend([
                    "Maintain current prevention strategies.",
                    "Continue surveillance to detect early signs of outbreaks.",
                    "Sustain public health education programs."
                ])

            trends = []
            for period, data in forecasts.get("forecasts", {}).items():
                if data.get("trend_direction") == "Increasing":
                    trends.append(period)

            if trends:
                recommendations.append(
                    f"WARNING: Increasing trend detected in {', '.join(trends)}. Enhance preparedness."
                )

            if forecasts.get("peak_prediction_period"):
                recommendations.append(
                    f"Plan resources around predicted peak period: {forecasts['peak_prediction_period']}"
                )

        if metrics and metrics.get("r2_score", 0) < 0.7:
            recommendations.append(
                "Consider collecting more diverse features to improve model accuracy."
            )

        recommendations.extend([
            "Implement indoor residual spraying before rainy seasons.",
            "Ensure early diagnosis and prompt treatment availability.",
            "Strengthen health information systems for real-time reporting."
        ])

        html = '<div class="section"><h2>Recommendations</h2><ul>'
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul></div>"
        return html

    def generate_html_report(self, dataset_info, metrics, forecasts, training_history):
        metrics_html = self._generate_metrics_table(metrics)
        charts_html = self._generate_charts_html(metrics, forecasts)
        forecast_html = self._generate_forecast_table(forecasts.get("forecasts", {})) if forecasts else ""
        recommendations_html = self._generate_recommendations(forecasts, metrics)

        dataset_summary = ""
        if dataset_info:
            dataset_summary = f"""
            <div class="section">
                <h2>Dataset Summary</h2>
                <table>
                    <thead><tr><th>Property</th><th>Value</th></tr></thead>
                    <tbody>
                        <tr><td>Total Samples</td><td>{dataset_info.get('total_samples', 'N/A')}</td></tr>
                        <tr><td>Features</td><td>{dataset_info.get('feature_count', 'N/A')}</td></tr>
                        <tr><td>Training Samples</td><td>{dataset_info.get('train_samples', 'N/A')}</td></tr>
                        <tr><td>Validation Samples</td><td>{dataset_info.get('val_samples', 'N/A')}</td></tr>
                        <tr><td>Test Samples</td><td>{dataset_info.get('test_samples', 'N/A')}</td></tr>
                        <tr><td>Sequence Length</td><td>{dataset_info.get('sequence_length', 'N/A')}</td></tr>
                    </tbody>
                </table>
            </div>
            """

        model_architecture = """
        <div class="section">
            <h2>Model Architecture</h2>
            <pre style="background:#f4f4f4;padding:15px;border-radius:8px;overflow-x:auto;">
Input Layer
    ↓
Simple RNN (64 units) + Dropout (0.2)
    ↓
LSTM (128 units) + Dropout (0.2)
    ↓
GRU (64 units)
    ↓
Dense (32 units, ReLU)
    ↓
Output Layer (1 unit, Linear)
            </pre>
            <p><strong>Optimizer:</strong> Adam (lr=0.001)</p>
            <p><strong>Loss Function:</strong> Mean Squared Error</p>
            <p><strong>Batch Size:</strong> 32 | <strong>Epochs:</strong> 100</p>
            <p><strong>Early Stopping:</strong> Enabled (patience=15)</p>
            <p><strong>ReduceLROnPlateau:</strong> Enabled</p>
        </div>
        """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Malaria Prediction Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #283593; margin: 25px 0 15px; padding-bottom: 8px; border-bottom: 2px solid #e8eaf6; }}
        h3 {{ color: #3949ab; margin: 15px 0 10px; }}
        .section {{ margin: 25px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #1a237e; color: #fff; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .chart-card {{ background: #fafafa; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .chart-card img {{ border-radius: 4px; }}
        .risk-low {{ color: #2e7d32; font-weight: bold; }}
        .risk-moderate {{ color: #f57f17; font-weight: bold; }}
        .risk-high {{ color: #e65100; font-weight: bold; }}
        .risk-critical {{ color: #c62828; font-weight: bold; }}
        ul {{ margin-left: 20px; }}
        li {{ margin: 8px 0; padding: 8px; background: #f5f5f5; border-radius: 4px; border-left: 4px solid #1a237e; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #e0e0e0; color: #666; font-size: 0.9em; }}
        @media (max-width: 600px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🦟 Malaria Prediction System Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Model:</strong> Hybrid RNN-LSTM-GRU</p>
        <hr>

        {dataset_summary}
        {model_architecture}
        {metrics_html}
        {charts_html}
        {forecast_html}
        {recommendations_html}

        <div class="footer">
            <p>Malaria Prediction System | Hybrid Deep Learning Model</p>
            <p>Generated automatically by the Malaria Prediction System</p>
        </div>
    </div>
</body>
</html>"""

        report_path = REPORTS_DIR / "malaria_prediction_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(report_path)

    def generate_csv_report(self, forecasts, metrics):
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Malaria Prediction Report"])
        writer.writerow([f"Generated: {datetime.now().isoformat()}"])
        writer.writerow([])

        if metrics:
            writer.writerow(["Performance Metrics"])
            for key, value in metrics.items():
                writer.writerow([key, value])
            writer.writerow([])

        if forecasts and "forecasts" in forecasts:
            writer.writerow(["Period", "Average", "Max", "Min", "Risk Level", "Trend", "Confidence"])
            for period, data in forecasts["forecasts"].items():
                writer.writerow([
                    period,
                    f"{data.get('average', 0):.2f}",
                    f"{data.get('max', 0):.2f}",
                    f"{data.get('min', 0):.2f}",
                    data.get("risk_level", ""),
                    data.get("trend_direction", ""),
                    f"{data.get('average_confidence', 0):.2%}"
                ])

        csv_path = REPORTS_DIR / "malaria_prediction_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write(output.getvalue())

        return str(csv_path)
