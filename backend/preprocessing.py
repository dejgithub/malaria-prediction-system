import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import logging
from pathlib import Path
import joblib

from config import (
    SEQUENCE_LENGTH, TEST_SPLIT, VAL_SPLIT, RANDOM_STATE,
    TARGET_COLUMN, FEATURE_COLUMNS, SCALER_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.preprocessing_log = []

    def load_data(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self._log(f"Loaded dataset: {file_path}, Shape: {df.shape}")
        return df

    def inspect_data(self, df):
        info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "basic_stats": df.describe(include="all").to_dict()
        }
        self._log(f"Data inspection complete. Missing values: {sum(df.isnull().sum())}")
        return info

    def handle_missing_values(self, df, strategy="ffill"):
        initial_missing = df.isnull().sum().sum()
        if initial_missing == 0:
            self._log("No missing values found.")
            return df

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["object", "category"]:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                else:
                    if strategy == "ffill":
                        df[col] = df[col].ffill()
                    elif strategy == "bfill":
                        df[col] = df[col].bfill()
                    elif strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "interpolate":
                        df[col] = df[col].interpolate()

                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mean() if df[col].dtype.kind in "iuf" else 0)

        filled = initial_missing - df.isnull().sum().sum()
        self._log(f"Missing values handled: {filled} values filled using '{strategy}' strategy.")
        return df

    def remove_duplicates(self, df):
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        if removed > 0:
            self._log(f"Removed {removed} duplicate rows.")
        return df

    def detect_outliers_iqr(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_report = {}
        for col in columns:
            if col not in df.columns or df[col].dtype.kind not in "iuf":
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_report[col] = {
                "count": int(len(outliers)),
                "percentage": float(round(len(outliers) / len(df) * 100, 2)),
                "lower_bound": float(round(lower, 4)),
                "upper_bound": float(round(upper, 4))
            }
        self._log(f"Outlier detection complete. Checked {len(columns)} columns.")
        return outlier_report

    def encode_categorical(self, df):
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            self._log(f"Encoded categorical column: {col}")
        return df

    def create_sequences(self, data, seq_length=None):
        if seq_length is None:
            seq_length = SEQUENCE_LENGTH
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    def split_data(self, X, y):
        split_idx = int(len(X) * (1 - TEST_SPLIT - VAL_SPLIT))
        val_idx = int(len(X) * (1 - TEST_SPLIT))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:val_idx]
        y_val = y[split_idx:val_idx]
        X_test = X[val_idx:]
        y_test = y[val_idx:]

        self._log(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def prepare_dataset(self, df, date_column=None):
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.sort_values(by=date_column).reset_index(drop=True)
            self._log(f"Sorted data by date column: {date_column}")

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)

        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            self._log(f"Warning: Missing features: {missing_features}")

        df = self.encode_categorical(df)

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_report = self.detect_outliers_iqr(df, numerical_cols)
        df_numeric = df[numerical_cols].copy()

        scaled_data = self.scaler.fit_transform(df_numeric)
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

        X, y = self.create_sequences(scaled_data)
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y)

        joblib.dump(self.scaler, SCALER_PATH)
        self._log(f"Scaler saved to {SCALER_PATH}")

        dataset_info = {
            "original_shape": list(df.shape),
            "numerical_columns": numerical_cols,
            "total_samples": len(df),
            "sequence_length": SEQUENCE_LENGTH,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_count": scaled_data.shape[1],
            "outlier_report": outlier_report,
            "available_features": available_features,
            "missing_features": missing_features
        }

        return X_train, y_train, X_val, y_val, X_test, y_test, dataset_info

    def get_preprocessing_log(self):
        return self.preprocessing_log

    def _log(self, message):
        logger.info(message)
        self.preprocessing_log.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

    def inverse_scale(self, scaled_values):
        dummy = np.zeros((len(scaled_values), self.scaler.scale_.shape[0]))
        dummy[:, 0] = scaled_values.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]

    def inverse_scale_features(self, scaled_matrix):
        return self.scaler.inverse_transform(scaled_matrix)
