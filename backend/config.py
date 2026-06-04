import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
DATASETS_DIR = BASE_DIR / "datasets"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "malaria_hybrid_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
HISTORY_PATH = MODEL_DIR / "training_history.json"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
LOSS_CURVE_PATH = MODEL_DIR / "loss_curves.png"
ACCURACY_CURVE_PATH = MODEL_DIR / "accuracy_curves.png"
PREDICTION_PLOT_PATH = MODEL_DIR / "actual_vs_predicted.png"

SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

RANDOM_STATE = 42

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024

TARGET_COLUMN = "Malaria Cases"
FEATURE_COLUMNS = [
    "Malaria Cases", "Malaria Deaths", "Temperature", "Rainfall",
    "Humidity", "Population", "Health Facility Coverage", "Environmental Factors"
]

RISK_THRESHOLDS = {
    "Low": 0.25,
    "Moderate": 0.50,
    "High": 0.75,
    "Critical": 1.0
}

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
