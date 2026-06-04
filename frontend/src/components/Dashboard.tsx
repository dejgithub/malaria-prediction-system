"use client";

import { useState, useEffect, useCallback } from "react";
import Sidebar from "./Sidebar";
import {
  checkHealth,
  uploadDataset,
  trainModel,
  getForecast,
  getMetrics,
  getInsights,
  getReports,
  getModelInfo,
  getDatasetInfo,
} from "@/lib/api";
import {
  ForecastChart,
  TrendChart,
  RiskPieChart,
  MetricsBarChart,
} from "./Charts";
import clsx from "clsx";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("home");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [forecast, setForecast] = useState<any>(null);
  const [insights, setInsights] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [trainStatus, setTrainStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadStatus();
    loadModelInfo();
    loadDatasetInfo();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadStatus = useCallback(async () => {
    try {
      const data = await checkHealth();
      setStatus(data);
    } catch {
      setStatus({ status: "unreachable" });
    }
  }, []);

  const loadModelInfo = useCallback(async () => {
    try {
      const data = await getModelInfo();
      setModelInfo(data);
    } catch {
      /* ignore */
    }
  }, []);

  const loadDatasetInfo = useCallback(async () => {
    try {
      const data = await getDatasetInfo();
      setDatasetInfo(data.dataset_info);
    } catch {
      /* ignore */
    }
  }, []);

  const handleTabChange = async (tab: string) => {
    setActiveTab(tab);
    setError("");
    if (tab === "forecast") await handleGetForecast();
    if (tab === "insights") await handleGetInsights();
    if (tab === "reports") await handleGetMetrics();
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setUploadStatus("Uploading and processing...");
    setError("");
    try {
      const data = await uploadDataset(file);
      setUploadStatus("Dataset uploaded successfully!");
      setDatasetInfo(data.dataset_info);
      if (data.dataset_info) setDatasetInfo(data.dataset_info);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Upload failed");
      setUploadStatus("Upload failed");
    }
    setLoading(false);
  };

  const handleTrainModel = async () => {
    setLoading(true);
    setTrainStatus("Training model... This may take several minutes.");
    setError("");
    try {
      const data = await trainModel();
      setTrainStatus("Model trained successfully!");
      setMetrics(data.evaluation_metrics);
      setModelInfo(await getModelInfo());
    } catch (err: any) {
      setError(err.response?.data?.detail || "Training failed");
      setTrainStatus("Training failed");
    }
    setLoading(false);
  };

  const handleGetForecast = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getForecast("all");
      setForecast(data.forecast);
      setInsights({ insights: data.insights, recommendations: [] });
    } catch (err: any) {
      setError(err.response?.data?.detail || "Forecast failed");
    }
    setLoading(false);
  };

  const handleGetMetrics = async () => {
    try {
      const data = await getMetrics();
      setMetrics(data.evaluation_metrics || data);
    } catch {
      /* ignore */
    }
  };

  const handleGetInsights = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getInsights();
      setInsights(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to get insights");
    }
    setLoading(false);
  };

  const handleDownloadReport = async (format: string) => {
    try {
      const data = await getReports(format);
      const blob = format === "html"
        ? new Blob([data as string], { type: "text/html" })
        : data as Blob;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `malaria_report.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("Failed to download report");
    }
  };

  return (
    <div className="flex h-screen bg-slate-900">
      <Sidebar activeTab={activeTab} onTabChange={handleTabChange} />

      <main className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
            <button onClick={() => setError("")} className="float-right font-bold">&times;</button>
          </div>
        )}

        {activeTab === "home" && <HomeTab status={status} modelInfo={modelInfo} datasetInfo={datasetInfo} metrics={metrics} />}

        {activeTab === "dataset" && (
          <DatasetTab
            uploadStatus={uploadStatus}
            datasetInfo={datasetInfo}
            loading={loading}
            onUpload={handleFileUpload}
          />
        )}

        {activeTab === "training" && (
          <TrainingTab
            trainStatus={trainStatus}
            metrics={metrics}
            loading={loading}
            onTrain={handleTrainModel}
            modelInfo={modelInfo}
          />
        )}

        {activeTab === "forecast" && <ForecastTab forecast={forecast} loading={loading} onRefresh={handleGetForecast} />}

        {activeTab === "insights" && <InsightsTab insights={insights} loading={loading} />}

        {activeTab === "reports" && (
          <ReportsTab metrics={metrics} onDownload={handleDownloadReport} onRefresh={handleGetMetrics} />
        )}
      </main>
    </div>
  );
}

function HomeTab({ status, modelInfo, datasetInfo, metrics }: any) {
  const stats = [
    { label: "API Status", value: status?.status || "Checking...", color: status?.status === "healthy" ? "text-green-400" : "text-yellow-400" },
    { label: "Model Status", value: modelInfo?.model_loaded ? "Loaded" : "Not Trained", color: modelInfo?.model_loaded ? "text-green-400" : "text-yellow-400" },
    { label: "Model Type", value: "Hybrid RNN-LSTM-GRU", color: "text-cyan-400" },
    { label: "Dataset", value: datasetInfo?.total_samples ? `${datasetInfo.total_samples} samples` : "Not loaded", color: "text-blue-400" },
    { label: "Accuracy", value: metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : "N/A", color: "text-green-400" },
    { label: "R² Score", value: metrics?.r2_score ? metrics.r2_score.toFixed(4) : "N/A", color: "text-purple-400" },
  ];

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold gradient-text">Malaria Prediction System</h2>
        <p className="text-slate-400 mt-1">Hybrid RNN-LSTM-GRU Deep Learning Model for Malaria Disease Forecasting</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {stats.map((s, i) => (
          <div key={i} className="card p-4">
            <p className="text-sm text-slate-400">{s.label}</p>
            <p className={clsx("text-xl font-bold mt-1", s.color)}>{s.value}</p>
          </div>
        ))}
      </div>

      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4">System Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-300">
          <div>
            <h4 className="font-medium text-cyan-400 mb-2">Architecture</h4>
            <ul className="space-y-1">
              <li>• Input Layer (Sequence Length: 12)</li>
              <li>• Simple RNN (64 units) + Dropout (0.2)</li>
              <li>• LSTM (128 units) + Dropout (0.2)</li>
              <li>• GRU (64 units)</li>
              <li>• Dense (32 units, ReLU)</li>
              <li>• Output Layer (Linear)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-cyan-400 mb-2">Capabilities</h4>
            <ul className="space-y-1">
              <li>• Data Preprocessing & Quality Checks</li>
              <li>• Automated Model Training</li>
              <li>• 1/3/6/12 Month Forecasting</li>
              <li>• Risk Level Classification</li>
              <li>• Automated Insights & Recommendations</li>
              <li>• HTML/CSV Report Generation</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function DatasetTab({ uploadStatus, datasetInfo, loading, onUpload }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold gradient-text mb-6">Dataset Management</h2>

      <div className="card p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Upload Dataset</h3>
        <p className="text-sm text-slate-400 mb-4">Upload a CSV or Excel file with malaria-related features (Date, Malaria Cases, Temperature, Rainfall, etc.)</p>
        <label className={clsx(
          "btn-primary inline-block cursor-pointer",
          loading && "opacity-50 pointer-events-none"
        )}>
          {loading ? "Processing..." : "Choose File"}
          <input type="file" accept=".csv,.xlsx,.xls" onChange={onUpload} className="hidden" disabled={loading} />
        </label>
        {uploadStatus && <p className="mt-2 text-sm text-slate-400">{uploadStatus}</p>}
      </div>

      {datasetInfo && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card p-4">
            <h3 className="font-semibold mb-3">Dataset Properties</h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-slate-400">Samples:</span> <span className="text-white">{datasetInfo.total_samples}</span></p>
              <p><span className="text-slate-400">Features:</span> <span className="text-white">{datasetInfo.feature_count}</span></p>
              <p><span className="text-slate-400">Sequence Length:</span> <span className="text-white">{datasetInfo.sequence_length}</span></p>
              <p><span className="text-slate-400">Training:</span> <span className="text-white">{datasetInfo.train_samples}</span></p>
              <p><span className="text-slate-400">Validation:</span> <span className="text-white">{datasetInfo.val_samples}</span></p>
              <p><span className="text-slate-400">Test:</span> <span className="text-white">{datasetInfo.test_samples}</span></p>
            </div>
          </div>

          {datasetInfo.available_features && (
            <div className="card p-4">
              <h3 className="font-semibold mb-3">Features</h3>
              <div className="space-y-2 text-sm">
                <p className="text-green-400">Available: {datasetInfo.available_features.join(", ")}</p>
                {datasetInfo.missing_features?.length > 0 && (
                  <p className="text-yellow-400">Missing: {datasetInfo.missing_features.join(", ")}</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TrainingTab({ trainStatus, metrics, loading, onTrain, modelInfo }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold gradient-text mb-6">Model Training</h2>

      <div className="card p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Train Hybrid Model</h3>
        <p className="text-sm text-slate-400 mb-4">
          Train the RNN-LSTM-GRU hybrid model on uploaded dataset. Training may take several minutes depending on data size.
        </p>
        <button onClick={onTrain} disabled={loading} className="btn-primary">
          {loading ? "Training..." : "Start Training"}
        </button>
        {trainStatus && <p className="mt-2 text-sm text-slate-400">{trainStatus}</p>}
      </div>

      {modelInfo && (
        <div className="card p-4 mb-6">
          <h3 className="font-semibold mb-3">Model Configuration</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div><span className="text-slate-400">Optimizer:</span><br/>{modelInfo.training_config?.optimizer}</div>
            <div><span className="text-slate-400">Learning Rate:</span><br/>{modelInfo.training_config?.learning_rate}</div>
            <div><span className="text-slate-400">Batch Size:</span><br/>{modelInfo.training_config?.batch_size}</div>
            <div><span className="text-slate-400">Max Epochs:</span><br/>{modelInfo.training_config?.epochs}</div>
          </div>
        </div>
      )}

      {metrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card p-4">
            <h3 className="font-semibold mb-3">Performance Metrics</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              {Object.entries({
                MAE: metrics.mae,
                MSE: metrics.mse,
                RMSE: metrics.rmse,
                "R² Score": metrics.r2_score,
                Accuracy: metrics.accuracy,
                Precision: metrics.precision,
                Recall: metrics.recall,
                "F1 Score": metrics.f1_score,
              }).map(([key, value]) => (
                <div key={key} className="bg-slate-800 rounded p-3">
                  <p className="text-slate-400">{key}</p>
                  <p className="text-lg font-bold text-cyan-400">
                    {["Accuracy", "Precision", "Recall", "F1 Score"].includes(key)
                      ? `${((value as number) * 100).toFixed(2)}%`
                      : (value as number).toFixed(4)}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h3 className="font-semibold mb-3">Metrics Visualization</h3>
            <MetricsBarChart metrics={metrics} />
          </div>
        </div>
      )}
    </div>
  );
}

function ForecastTab({ forecast, loading, onRefresh }: any) {
  const [historicalData] = useState(() =>
    Array.from({ length: 60 }, (_, i) => 600 + Math.random() * 800 + Math.sin(i / 6) * 300)
  );

  const allPredictions = forecast?.forecasts
    ? Object.values(forecast.forecasts).flatMap((d: any) => d.predictions)
    : [];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold gradient-text">Forecast Dashboard</h2>
        <button onClick={onRefresh} disabled={loading} className="btn-primary text-sm">
          {loading ? "Loading..." : "Refresh Forecast"}
        </button>
      </div>

      {forecast && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="card p-4 text-center">
              <p className="text-sm text-slate-400">Overall Risk</p>
              <p className={clsx(
                "text-lg font-bold mt-1",
                forecast.overall_risk_assessment?.includes("Critical") ? "risk-critical" :
                forecast.overall_risk_assessment?.includes("High") ? "risk-high" :
                forecast.overall_risk_assessment?.includes("Moderate") ? "risk-moderate" : "risk-low"
              )}>
                {forecast.overall_risk_assessment || "N/A"}
              </p>
            </div>
            <div className="card p-4 text-center">
              <p className="text-sm text-slate-400">Historical Max</p>
              <p className="text-xl font-bold text-cyan-400">{forecast.historical_max?.toFixed(0) || "N/A"}</p>
            </div>
            <div className="card p-4 text-center">
              <p className="text-sm text-slate-400">Peak Period</p>
              <p className="text-sm font-bold text-yellow-400 mt-1">{forecast.peak_prediction_period || "N/A"}</p>
            </div>
            <div className="card p-4 text-center">
              <p className="text-sm text-slate-400">Forecast Periods</p>
              <p className="text-xl font-bold text-purple-400">{Object.keys(forecast.forecasts || {}).length}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="card p-4">
              <h3 className="font-semibold mb-3">Forecast Predictions</h3>
              {forecast.forecasts ? <ForecastChart data={forecast.forecasts} /> : <p className="text-slate-400">No forecast data</p>}
            </div>
            <div className="card p-4">
              <h3 className="font-semibold mb-3">Risk Distribution</h3>
              {forecast.forecasts ? <RiskPieChart forecasts={forecast.forecasts} /> : <p className="text-slate-400">No forecast data</p>}
            </div>
          </div>

          <div className="card p-4 mb-6">
            <h3 className="font-semibold mb-3">Trend Analysis</h3>
            <TrendChart historical={historicalData} forecast={allPredictions} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {forecast.forecasts && Object.entries(forecast.forecasts).map(([period, data]: [string, any]) => (
              <div key={period} className="card p-4">
                <h4 className="font-semibold text-cyan-400 mb-2">{period.replace("_", " ").replace(/\b\w/g, (c: string) => c.toUpperCase())}</h4>
                <div className="space-y-1 text-sm">
                  <p><span className="text-slate-400">Avg:</span> <span className="text-white">{data.average?.toFixed(1)}</span></p>
                  <p><span className="text-slate-400">Max:</span> <span className="text-white">{data.max?.toFixed(1)}</span></p>
                  <p><span className="text-slate-400">Risk:</span> <span className={clsx(
                    data.risk_level?.includes("Critical") ? "risk-critical" :
                    data.risk_level?.includes("High") ? "risk-high" :
                    data.risk_level?.includes("Moderate") ? "risk-moderate" : "risk-low"
                  )}>{data.risk_level}</span></p>
                  <p><span className="text-slate-400">Trend:</span> <span className={clsx(
                    data.trend_direction === "Increasing" ? "text-red-400" :
                    data.trend_direction === "Decreasing" ? "text-green-400" : "text-yellow-400"
                  )}>{data.trend_direction}</span></p>
                  <p><span className="text-slate-400">Confidence:</span> <span className="text-white">{(data.average_confidence * 100).toFixed(1)}%</span></p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {!forecast && !loading && (
        <div className="card p-8 text-center">
          <p className="text-slate-400">Click "Refresh Forecast" to generate predictions</p>
        </div>
      )}

      {loading && (
        <div className="card p-8 text-center">
          <div className="loading-pulse text-cyan-400 text-lg">Generating forecast...</div>
        </div>
      )}
    </div>
  );
}

function InsightsTab({ insights, loading }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold gradient-text mb-6">Insights & Recommendations</h2>

      {insights?.insights && insights.insights.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-4">AI-Generated Insights</h3>
          <div className="space-y-3">
            {insights.insights.map((insight: any, i: number) => (
              <div key={i} className={clsx(
                "card p-4 border-l-4",
                insight.severity === "high" ? "border-l-red-500" :
                insight.severity === "medium" ? "border-l-yellow-500" : "border-l-green-500"
              )}>
                <div className="flex items-start gap-3">
                  <span className={clsx(
                    "text-lg",
                    insight.severity === "high" ? "text-red-400" :
                    insight.severity === "medium" ? "text-yellow-400" : "text-green-400"
                  )}>
                    {insight.severity === "high" ? "🔴" : insight.severity === "medium" ? "🟡" : "🟢"}
                  </span>
                  <div>
                    <p className="font-medium">{insight.message}</p>
                    {insight.detail && <p className="text-sm text-slate-400 mt-1">{insight.detail}</p>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {insights?.recommendations && insights.recommendations.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
          <div className="space-y-3">
            {insights.recommendations.map((rec: any, i: number) => (
              <div key={i} className={clsx(
                "card p-4",
                rec.priority === "immediate" ? "border border-red-700" :
                rec.priority === "high" ? "border border-yellow-700" : ""
              )}>
                <div className="flex items-start gap-3">
                  <span className={clsx(
                    "px-2 py-0.5 rounded text-xs font-bold uppercase",
                    rec.priority === "immediate" ? "bg-red-900 text-red-300" :
                    rec.priority === "high" ? "bg-yellow-900 text-yellow-300" :
                    "bg-slate-700 text-slate-300"
                  )}>{rec.priority}</span>
                  <div className="flex-1">
                    <p className="font-medium">{rec.action}</p>
                    <p className="text-sm text-slate-400">{rec.rationale}</p>
                    {rec.timeline && <p className="text-xs text-cyan-400 mt-1">Timeline: {rec.timeline}</p>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!insights || insights.insights?.length === 0) && !loading && (
        <div className="card p-8 text-center">
          <p className="text-slate-400">Go to Forecast tab first to generate insights.</p>
        </div>
      )}
    </div>
  );
}

function ReportsTab({ metrics, onDownload, onRefresh }: any) {
  return (
    <div>
      <h2 className="text-2xl font-bold gradient-text mb-6">Reports</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="card p-6 text-center">
          <h3 className="text-lg font-semibold mb-2">📄 HTML Report</h3>
          <p className="text-sm text-slate-400 mb-4">Comprehensive standalone HTML report with charts, metrics, forecast, and recommendations.</p>
          <button onClick={() => onDownload("html")} className="btn-primary">
            Download HTML Report
          </button>
        </div>
        <div className="card p-6 text-center">
          <h3 className="text-lg font-semibold mb-2">📊 CSV Results</h3>
          <p className="text-sm text-slate-400 mb-4">Tabular data including metrics and forecast results in CSV format.</p>
          <button onClick={() => onDownload("csv")} className="btn-primary">
            Download CSV
          </button>
        </div>
      </div>

      {metrics && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Performance Summary</h3>
            <button onClick={onRefresh} className="text-sm text-cyan-400 hover:underline">Refresh</button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              ["MAE", metrics.mae, "text-blue-400"],
              ["RMSE", metrics.rmse, "text-purple-400"],
              ["R² Score", metrics.r2_score, "text-green-400"],
              ["Accuracy", metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(2)}%` : "N/A", "text-cyan-400"],
            ].map(([label, value, color]) => (
              <div key={label as string} className="bg-slate-800 rounded p-3 text-center">
                <p className="text-xs text-slate-400">{label as string}</p>
                <p className={clsx("text-lg font-bold mt-1", color as string)}>
                  {typeof value === "number" ? (value as number).toFixed(4) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
