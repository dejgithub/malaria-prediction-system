import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import logging
from datetime import datetime

from config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_PATH,
    HISTORY_PATH, METRICS_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HybridMalariaModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.metrics = {}

    def build_model(self):
        inputs = keras.Input(shape=self.input_shape, name="input_layer")

        x = layers.SimpleRNN(64, return_sequences=True, activation="tanh", name="rnn_layer")(inputs)
        x = layers.Dropout(0.2, name="dropout_1")(x)

        x = layers.LSTM(128, return_sequences=True, activation="tanh", name="lstm_layer")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)

        x = layers.GRU(64, return_sequences=False, activation="tanh", name="gru_layer")(x)

        x = layers.Dense(32, activation="relu", name="dense_layer")(x)

        outputs = layers.Dense(1, activation="linear", name="output_layer")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_RNN_LSTM_GRU_Malaria")

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )

        logger.info(f"Model built with input shape {self.input_shape}")
        self.model.summary(print_fn=logger.info)
        return self.model

    def get_callbacks(self):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode="min"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1,
                mode="min"
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
                mode="min"
            ),
            keras.callbacks.CSVLogger(
                filename=str(MODEL_PATH.parent / "training_log.csv"),
                append=True
            )
        ]
        return callbacks

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()

        callbacks = self.get_callbacks()

        logger.info(f"Starting training: {EPOCHS} epochs, batch size {BATCH_SIZE}")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        self.save_history()
        logger.info("Training completed successfully")
        return self.history

    def predict(self, X):
        if self.model is None:
            self.load_model()
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            self.load_model()

        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.predict(X_test)

        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            r2_score, accuracy_score, precision_score,
            recall_score, f1_score
        )

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        y_test_binary = (y_test > np.median(y_test)).astype(int)
        y_pred_binary = (y_pred.flatten() > np.median(y_test)).astype(int)

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

        self.metrics = {
            "loss": float(loss),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "test_samples": len(y_test)
        }

        self.save_metrics()
        logger.info(f"Evaluation metrics: {json.dumps(self.metrics, indent=2)}")
        return self.metrics, y_pred

    def save_history(self):
        if self.history:
            history_dict = {
                "epochs": len(self.history.history["loss"]),
                "loss": [float(v) for v in self.history.history["loss"]],
                "val_loss": [float(v) for v in self.history.history["val_loss"]],
                "mae": [float(v) for v in self.history.history["mae"]],
                "val_mae": [float(v) for v in self.history.history["val_mae"]]
            }
            with open(HISTORY_PATH, "w") as f:
                json.dump(history_dict, f, indent=2)
            logger.info(f"Training history saved to {HISTORY_PATH}")

    def save_metrics(self):
        with open(METRICS_PATH, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {METRICS_PATH}")

    def load_model(self):
        if MODEL_PATH.exists():
            self.model = keras.models.load_model(str(MODEL_PATH))
            logger.info(f"Model loaded from {MODEL_PATH}")
            return True
        logger.warning(f"No saved model found at {MODEL_PATH}")
        return False

    def save_model(self):
        self.model.save(str(MODEL_PATH))
        logger.info(f"Model saved to {MODEL_PATH}")
