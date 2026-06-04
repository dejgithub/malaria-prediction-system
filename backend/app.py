import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MODEL_PATH, SCALER_PATH, HISTORY_PATH, METRICS_PATH,
    SEQUENCE_LENGTH, CORS_ORIGINS, API_RATE_LIMIT,
    FEATURE_COLUMNS, DATASETS_DIR, REPORTS_DIR,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE
)
from preprocessing import DataPreprocessor
from model_builder import HybridMalariaModel
from forecasting import ForecastEngine
from visualization import VisualizationEngine
from report_generator import ReportGenerator
from insights import InsightsEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

preprocessor = DataPreprocessor()
model_builder = None
forecast_engine = None
visualizer = VisualizationEngine()
report_gen = ReportGenerator()

dataset_info = None
training_history = None
evaluation_metrics = None
forecast_results = None
insights_engine = None
last_sequence_data = None
historical_data_values = None
scaled_data = None
all_features = FEATURE_COLUMNS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_builder, forecast_engine
    logger.info("Starting Malaria Prediction System...")
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        logger.info("Loading existing model and scaler...")
        try:
            scaler = joblib.load(SCALER_PATH)
            n_features = scaler.n_features_in_
            model_builder = HybridMalariaModel(input_shape=(SEQUENCE_LENGTH, n_features))
            loaded = model_builder.load_model()
            forecast_engine = ForecastEngine(model_builder.model, scaler, FEATURE_COLUMNS)
            if loaded:
                logger.info("Model and scaler loaded successfully")
            else:
                logger.warning("Failed to load model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    else:
        logger.info("No existing model found. Ready for training.")
        input_shape = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        model_builder = HybridMalariaModel(input_shape=input_shape)
    yield
    logger.info("Shutting down Malaria Prediction System...")


app = FastAPI(
    title="Malaria Prediction System API",
    description="Hybrid RNN-LSTM-GRU Model for Malaria Disease Prediction",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = CORS_ORIGINS.split(",") if CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False if cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def root(request: Request):
    return {
        "service": "Malaria Prediction System",
        "version": "2.0.0",
        "status": "operational",
        "model_type": "Hybrid RNN-LSTM-GRU",
        "endpoints": {
            "health": "/health",
            "upload_dataset": "/upload-dataset",
            "train_model": "/train-model",
            "predict": "/predict",
            "forecast": "/forecast",
            "metrics": "/metrics",
            "reports": "/reports",
            "insights": "/insights",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def health_check(request: Request):
    model_loaded = MODEL_PATH.exists() and SCALER_PATH.exists()
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH) if model_loaded else None,
        "api_version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload-dataset")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    global preprocessor, dataset_info, all_features

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file format. Allowed: {ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    file_path = DATASETS_DIR / f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        df = preprocessor.load_data(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")

    data_info = preprocessor.inspect_data(df)
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

    X_train, y_train, X_val, y_val, X_test, y_test, ds_info = preprocessor.prepare_dataset(df)
    dataset_info = ds_info
    dataset_info["file_name"] = file.filename
    dataset_info["upload_time"] = datetime.now().isoformat()
    dataset_info["inspection"] = data_info
    dataset_info["available_features"] = available_features
    dataset_info["missing_features"] = missing_features

    global scaled_data
    scaled_data = {
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "total_sequences": len(X_train) + len(X_val) + len(X_test)
    }
    logger.info(f"Dataset uploaded and processed: {file.filename}")
    return {
        "message": "Dataset uploaded and processed successfully",
        "dataset_info": dataset_info,
        "preprocessing_log": preprocessor.get_preprocessing_log()[-5:]
    }


@app.post("/train-model")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def train_model(request: Request):
    global model_builder, forecast_engine, training_history, evaluation_metrics, scaled_data, preprocessor

    if not hasattr(preprocessor, "scaler") or not SCALER_PATH.exists():
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    try:
        df_paths = list(DATASETS_DIR.glob("dataset_*"))
        if not df_paths:
            raise HTTPException(status_code=400, detail="No dataset found. Please upload a dataset.")

        latest_dataset = max(df_paths, key=os.path.getctime)
        df = preprocessor.load_data(str(latest_dataset))
        X_train, y_train, X_val, y_val, X_test, y_test, ds_info = preprocessor.prepare_dataset(df)

        global dataset_info
        dataset_info = ds_info

        input_shape = (X_train.shape[1], X_train.shape[2])
        model_builder = HybridMalariaModel(input_shape=input_shape)
        model_builder.build_model()

        logger.info("Starting model training...")
        history = model_builder.train(X_train, y_train, X_val, y_val)

        training_history = {
            "epochs": len(history.history["loss"]),
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "mae": [float(v) for v in history.history["mae"]],
            "val_mae": [float(v) for v in history.history["val_mae"]],
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_mae": float(history.history["mae"][-1]),
            "final_val_mae": float(history.history["val_mae"][-1])
        }

        metrics, y_pred = model_builder.evaluate(X_test, y_test)
        evaluation_metrics = metrics

        scaler = joblib.load(SCALER_PATH)
        forecast_engine = ForecastEngine(model_builder.model, scaler, all_features)

        vis_results = visualizer.generate_all_plots(
            {"loss": training_history["loss"], "val_loss": training_history["val_loss"],
             "mae": training_history["mae"], "val_mae": training_history["val_mae"]},
            y_test, y_pred
        )

        global last_sequence_data, historical_data_values
        last_sequence_data = X_test[-1]
        all_scaled = np.vstack([X_train, X_val, X_test])
        dummy = np.zeros((all_scaled.shape[0] * all_scaled.shape[1], scaler.n_features_in_))
        dummy[:, 0] = all_scaled[:, :, 0].flatten()
        historical_data_values = scaler.inverse_transform(dummy)[:, 0]

        model_builder.save_model()

        return {
            "message": "Model trained successfully",
            "training_history": training_history,
            "evaluation_metrics": metrics,
            "visualizations": vis_results,
            "dataset_info": dataset_info
        }

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def predict(request: Request, data: dict = None):
    global model_builder, forecast_engine, last_sequence_data

    if model_builder is None or model_builder.model is None:
        if not model_builder.load_model():
            raise HTTPException(status_code=400, detail="No trained model available. Train the model first.")

    if forecast_engine is None:
        scaler = joblib.load(SCALER_PATH)
        n_features = scaler.n_features_in_
        forecast_engine = ForecastEngine(model_builder.model, scaler, all_features)

    if data and "features" in data:
        input_features = np.array(data["features"], dtype=np.float32)
        if input_features.ndim == 1:
            input_features = input_features.reshape(1, -1)
        if input_features.shape[1] != SEQUENCE_LENGTH:
            input_features = input_features.T
        scaler = joblib.load(SCALER_PATH)
        n_features = scaler.n_features_in_
        if input_features.shape != (1, SEQUENCE_LENGTH, n_features):
            input_features = np.tile(input_features, (1, SEQUENCE_LENGTH, 1))
        pred_scaled = model_builder.model.predict(input_features, verbose=0)
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = pred_scaled[0, 0]
        pred_actual = float(scaler.inverse_transform(dummy)[0, 0])
        return {"prediction": max(0, pred_actual), "input_shape": list(input_features.shape)}

    if last_sequence_data is None:
        raise HTTPException(status_code=400, detail="No sequence data available. Upload dataset and train first.")

    preds, confs = forecast_engine.generate_forecast(last_sequence_data, 1)
    return {"prediction": preds[0], "confidence": confs[0], "unit": "monthly malaria cases"}


@app.get("/forecast")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_forecast(request: Request, periods: str = "all"):
    global forecast_engine, forecast_results, last_sequence_data, historical_data_values

    if forecast_engine is None or last_sequence_data is None:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                scaler = joblib.load(SCALER_PATH)
                n_features = scaler.n_features_in_
                model_builder = HybridMalariaModel(input_shape=(SEQUENCE_LENGTH, n_features))
                model_builder.load_model()
                forecast_engine = ForecastEngine(model_builder.model, scaler, all_features)
                dummy_seq = np.zeros((SEQUENCE_LENGTH, n_features))
                last_sequence_data = dummy_seq
                historical_data_values = np.array([])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot initialize forecast: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No model available. Train the model first.")

    forecast_results = forecast_engine.generate_forecast_report(last_sequence_data, historical_data_values)

    global insights_engine
    insights_engine = InsightsEngine(forecast_results, historical_data_values)

    if periods != "all":
        valid_periods = periods.split(",")
        forecast_results["forecasts"] = {
            k: v for k, v in forecast_results["forecasts"].items()
            if k in valid_periods
        }

    return {
        "forecast": forecast_results,
        "insights": insights_engine.generate_insights()[:5]
    }


@app.get("/insights")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_insights(request: Request):
    global insights_engine, forecast_results

    if insights_engine is None:
        if forecast_results:
            insights_engine = InsightsEngine(forecast_results, historical_data_values)
        else:
            return {
                "insights": [{"type": "info", "severity": "low", "message": "Generate a forecast first to see insights."}],
                "recommendations": []
            }

    return {
        "insights": insights_engine.generate_insights(),
        "recommendations": insights_engine.generate_recommendations()
    }


@app.get("/metrics")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_metrics(request: Request):
    global evaluation_metrics, training_history

    if evaluation_metrics is None and METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            evaluation_metrics = json.load(f)

    if training_history is None and HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            training_history = json.load(f)

    return {
        "evaluation_metrics": evaluation_metrics or {},
        "training_history": training_history or {},
        "model_exists": MODEL_PATH.exists()
    }


@app.get("/reports")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_reports(request: Request, format: str = "html"):
    global dataset_info, evaluation_metrics, forecast_results, training_history

    if format == "html":
        report_path = report_gen.generate_html_report(
            dataset_info, evaluation_metrics, forecast_results, training_history
        )
        return FileResponse(report_path, media_type="text/html", filename="malaria_prediction_report.html")
    elif format == "csv":
        csv_path = report_gen.generate_csv_report(forecast_results, evaluation_metrics)
        return FileResponse(csv_path, media_type="text/csv", filename="malaria_prediction_results.csv")
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'html' or 'csv'.")


@app.get("/dataset-info")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_dataset_info(request: Request):
    if dataset_info is None:
        return {"message": "No dataset loaded. Upload a dataset first.", "dataset_info": None}
    return {"dataset_info": dataset_info}


@app.get("/model-info")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_model_info(request: Request):
    n_features = len(all_features)
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            n_features = scaler.n_features_in_
        except:
            pass
    return {
        "model_type": "Hybrid RNN-LSTM-GRU",
        "architecture": {
            "input_shape": f"(batch, {SEQUENCE_LENGTH}, {n_features})",
            "layers": [
                {"SimpleRNN": {"units": 64, "return_sequences": True, "activation": "tanh"}},
                {"Dropout": {"rate": 0.2}},
                {"LSTM": {"units": 128, "return_sequences": True, "activation": "tanh"}},
                {"Dropout": {"rate": 0.2}},
                {"GRU": {"units": 64, "return_sequences": False, "activation": "tanh"}},
                {"Dense": {"units": 32, "activation": "relu"}},
                {"Dense": {"units": 1, "activation": "linear"}}
            ]
        },
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 7
        },
        "model_loaded": MODEL_PATH.exists()
    }


@app.get("/visualizations/{viz_name}")
@limiter.limit(f"{API_RATE_LIMIT}/minute")
async def get_visualization(request: Request, viz_name: str):
    viz_map = {
        "confusion_matrix": CONFUSION_MATRIX_PATH,
        "loss_curves": LOSS_CURVE_PATH,
        "accuracy_curves": ACCURACY_CURVE_PATH,
        "actual_vs_predicted": PREDICTION_PLOT_PATH
    }
    if viz_name not in viz_map:
        raise HTTPException(status_code=404, detail=f"Visualization not found. Available: {list(viz_map.keys())}")
    viz_path = viz_map[viz_name]
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not generated yet. Train the model first.")
    return FileResponse(str(viz_path), media_type="image/png")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
