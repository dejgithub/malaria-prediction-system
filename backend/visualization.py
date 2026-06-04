import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from config import (
    CONFUSION_MATRIX_PATH, LOSS_CURVE_PATH,
    ACCURACY_CURVE_PATH, PREDICTION_PLOT_PATH
)


class VisualizationEngine:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        if save_path is None:
            save_path = CONFUSION_MATRIX_PATH
        y_true_binary = (y_true > np.median(y_true)).astype(int)
        y_pred_binary = (y_pred.flatten() > np.median(y_true)).astype(int)

        cm = confusion_matrix(y_true_binary, y_pred_binary)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low Risk", "High Risk"],
                    yticklabels=["Low Risk", "High Risk"])
        ax.set_title("Confusion Matrix - Malaria Risk Classification")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(save_path)

    @staticmethod
    def plot_loss_curves(history, save_path=None):
        if save_path is None:
            save_path = LOSS_CURVE_PATH
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
        ax.set_title("Model Loss Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(save_path)

    @staticmethod
    def plot_accuracy_curves(history, save_path=None):
        if save_path is None:
            save_path = ACCURACY_CURVE_PATH
        if "mae" not in history:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history["mae"]) + 1)
        ax.plot(epochs, history["mae"], "b-", label="Training MAE", linewidth=2)
        ax.plot(epochs, history["val_mae"], "r-", label="Validation MAE", linewidth=2)
        ax.set_title("Model MAE Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MAE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(save_path)

    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred, save_path=None):
        if save_path is None:
            save_path = PREDICTION_PLOT_PATH
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(y_test, "b-", label="Actual", linewidth=2, alpha=0.7)
        axes[0].plot(y_pred.flatten(), "r-", label="Predicted", linewidth=2, alpha=0.7)
        axes[0].set_title("Actual vs Predicted (Time Series)")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Scaled Malaria Cases")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(y_test, y_pred.flatten(), alpha=0.6, s=30, c="#1a237e")
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
        axes[1].set_title("Actual vs Predicted (Scatter)")
        axes[1].set_xlabel("Actual Values")
        axes[1].set_ylabel("Predicted Values")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(save_path)

    @staticmethod
    def generate_all_plots(history, y_test, y_pred):
        results = {}
        results["confusion_matrix"] = VisualizationEngine.plot_confusion_matrix(y_test, y_pred)
        results["loss_curves"] = VisualizationEngine.plot_loss_curves(history)
        results["accuracy_curves"] = VisualizationEngine.plot_accuracy_curves(history)
        results["actual_vs_predicted"] = VisualizationEngine.plot_actual_vs_predicted(y_test, y_pred)
        return results
