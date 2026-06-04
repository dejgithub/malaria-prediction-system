"use client";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line, Bar, Pie } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const chartColors = {
  cyan: "rgba(0, 188, 212, 1)",
  cyanBg: "rgba(0, 188, 212, 0.1)",
  blue: "rgba(2, 119, 189, 1)",
  blueBg: "rgba(2, 119, 189, 0.1)",
  green: "rgba(74, 222, 128, 1)",
  yellow: "rgba(251, 191, 36, 1)",
  orange: "rgba(251, 146, 60, 1)",
  orangeBg: "rgba(251, 146, 60, 0.1)",
  red: "rgba(248, 113, 113, 1)",
};

const defaultOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: "#94a3b8" },
    },
  },
  scales: {
    x: {
      ticks: { color: "#64748b" },
      grid: { color: "rgba(51, 65, 85, 0.5)" },
    },
    y: {
      ticks: { color: "#64748b" },
      grid: { color: "rgba(51, 65, 85, 0.5)" },
    },
  },
};

export function ForecastChart({ data }: { data: Record<string, { predictions: number[]; risk_level: string }> }) {
  const labels: string[] = [];
  const values: number[] = [];
  const colors: string[] = [];

  Object.entries(data).forEach(([period, periodData]) => {
    const displayName = period.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
    periodData.predictions.forEach((_, idx) => {
      labels.push(`${displayName} (M${idx + 1})`);
      values.push(periodData.predictions[idx]);
      const risk = periodData.risk_level.toLowerCase();
      if (risk.includes("critical")) colors.push(chartColors.red);
      else if (risk.includes("high")) colors.push(chartColors.orange);
      else if (risk.includes("moderate")) colors.push(chartColors.yellow);
      else colors.push(chartColors.green);
    });
  });

  return (
    <div className="chart-container">
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: "Predicted Cases",
              data: values,
              backgroundColor: colors.map((c) => c.replace("1)", "0.7)")),
              borderColor: colors,
              borderWidth: 2,
              borderRadius: 4,
            },
          ],
        }}
        options={{
          ...defaultOptions,
          plugins: {
            ...defaultOptions.plugins,
            title: { display: true, text: "Forecast Predictions", color: "#f1f5f9" },
          },
        }}
      />
    </div>
  );
}

export function TrendChart({
  historical,
  forecast,
}: {
  historical?: number[];
  forecast?: number[];
}) {
  const hist = historical || [];
  const fore = forecast || [];
  const histLabels = hist.map((_, i) => `M${i + 1}`);
  const foreLabels = fore.map((_, i) => `F M${i + 1}`);

  return (
    <div className="chart-container">
      <Line
        data={{
          labels: [...histLabels, ...foreLabels],
          datasets: [
            {
              label: "Historical",
              data: [...hist, ...Array(fore.length).fill(null)],
              borderColor: chartColors.cyan,
              backgroundColor: chartColors.cyanBg,
              fill: true,
              tension: 0.4,
              pointRadius: 3,
            },
            {
              label: "Forecast",
              data: [...Array(hist.length).fill(null), ...fore],
              borderColor: chartColors.orange,
              backgroundColor: chartColors.orangeBg || "rgba(251, 146, 60, 0.1)",
              borderDash: [5, 5],
              fill: true,
              tension: 0.4,
              pointRadius: 4,
              pointBackgroundColor: chartColors.orange,
            },
          ],
        }}
        options={{
          ...defaultOptions,
          plugins: {
            ...defaultOptions.plugins,
            title: { display: true, text: "Historical & Forecast Trend", color: "#f1f5f9" },
          },
        }}
      />
    </div>
  );
}

export function RiskPieChart({ forecasts }: { forecasts: Record<string, { risk_level: string }> }) {
  const riskCounts: Record<string, number> = {};
  Object.values(forecasts).forEach((d) => {
    const risk = d.risk_level;
    riskCounts[risk] = (riskCounts[risk] || 0) + 1;
  });

  const labels = Object.keys(riskCounts);
  const data = Object.values(riskCounts);
  const colors = labels.map((l) => {
    if (l.includes("Critical")) return chartColors.red;
    if (l.includes("High")) return chartColors.orange;
    if (l.includes("Moderate")) return chartColors.yellow;
    return chartColors.green;
  });

  return (
    <div className="chart-container" style={{ maxHeight: 300 }}>
      <Pie
        data={{
          labels,
          datasets: [{ data, backgroundColor: colors, borderColor: "#1e293b", borderWidth: 2 }],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom", labels: { color: "#94a3b8", padding: 12 } },
            title: { display: true, text: "Risk Distribution", color: "#f1f5f9" },
          },
        }}
      />
    </div>
  );
}

export function MetricsBarChart({ metrics }: { metrics: Record<string, number> }) {
  const metricLabels: Record<string, string> = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1_score: "F1 Score",
    r2_score: "R² Score",
  };

  const labels: string[] = [];
  const values: number[] = [];
  Object.entries(metricLabels).forEach(([key, label]) => {
    if (metrics[key] !== undefined) {
      labels.push(label);
      values.push(metrics[key]);
    }
  });

  return (
    <div className="chart-container">
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: "Score",
              data: values,
              backgroundColor: [
                chartColors.green,
                chartColors.blue,
                chartColors.cyan,
                chartColors.yellow,
                chartColors.orange,
              ].map((c) => c.replace("1)", "0.7)")),
              borderColor: [
                chartColors.green,
                chartColors.blue,
                chartColors.cyan,
                chartColors.yellow,
                chartColors.orange,
              ],
              borderWidth: 2,
              borderRadius: 4,
            },
          ],
        }}
        options={{
          ...defaultOptions,
          indexAxis: "y" as const,
          plugins: {
            ...defaultOptions.plugins,
            title: { display: true, text: "Model Performance Metrics", color: "#f1f5f9" },
          },
        }}
      />
    </div>
  );
}

export function LossCurveChart({ history }: { history: { loss: number[]; val_loss: number[] } }) {
  const epochs = history.loss.map((_, i) => i + 1);

  return (
    <div className="chart-container">
      <Line
        data={{
          labels: epochs,
          datasets: [
            {
              label: "Training Loss",
              data: history.loss,
              borderColor: chartColors.cyan,
              backgroundColor: chartColors.cyanBg,
              fill: true,
              tension: 0.3,
              pointRadius: 1,
            },
            {
              label: "Validation Loss",
              data: history.val_loss,
              borderColor: chartColors.orange,
              backgroundColor: "rgba(251, 146, 60, 0.1)",
              fill: true,
              tension: 0.3,
              pointRadius: 1,
            },
          ],
        }}
        options={{
          ...defaultOptions,
          plugins: {
            ...defaultOptions.plugins,
            title: { display: true, text: "Training & Validation Loss", color: "#f1f5f9" },
          },
        }}
      />
    </div>
  );
}
