import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  timeout: 300000,
});

export const checkHealth = async () => {
  const { data } = await api.get("/health");
  return data;
};

export const uploadDataset = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/upload-dataset", formData, {
    timeout: 60000,
  });
  return data;
};

export const trainModel = async () => {
  const { data } = await api.post("/train-model", {}, { timeout: 600000 });
  return data;
};

export const getForecast = async (periods = "all") => {
  const { data } = await api.get(`/forecast?periods=${periods}`);
  return data;
};

export const getMetrics = async () => {
  const { data } = await api.get("/metrics");
  return data;
};

export const getInsights = async () => {
  const { data } = await api.get("/insights");
  return data;
};

export const getReports = async (format = "html") => {
  const { data } = await api.get(`/reports?format=${format}`, {
    responseType: format === "html" ? "text" : "blob",
  });
  return data;
};

export const getModelInfo = async () => {
  const { data } = await api.get("/model-info");
  return data;
};

export const getDatasetInfo = async () => {
  const { data } = await api.get("/dataset-info");
  return data;
};

export default api;
