import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.services.F1MetricsCallback import F1MetricsCallback
import torch


class PlotService:
    """
    Handles all visualization tasks for experiments.
    """

    def __init__(self, experiment_name: str, output_dir: Path, experiment_config: dict):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.config = experiment_config

    def plot_confusion_matrix(self, model, X_test, y_test, label_dict: dict):
        """
        Generate and save confusion matrix plot.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            label_dict: Label mapping dictionary
        """
        y_pred = model.predict(X_test, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='g',
            xticklabels=label_dict.keys(),
            yticklabels=label_dict.keys(),
            cmap='Blues'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.experiment_name}')
        plt.tight_layout()

        dataset = self.config.get("dataset_name", "unknown")
        save_path = self.output_dir / f"{self.experiment_name}_{dataset}_confusion_matrix.png"

        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Confusion matrix saved to {save_path}")

    def plot_f1_curves(self, f1_callback: F1MetricsCallback):
        """
        Plot training and validation F1 score curves.

        Args:
            f1_callback: Callback instance containing F1 scores
        """
        if not f1_callback.train_f1_scores or not f1_callback.val_f1_scores:
            print("No F1 scores available to plot.")
            return

        plt.figure(figsize=(10, 5))

        epochs = range(1, len(f1_callback.train_f1_scores) + 1)
        plt.plot(epochs, f1_callback.train_f1_scores,
                 label='Training F1', color='red', linestyle='-')
        plt.plot(epochs, f1_callback.val_f1_scores,
                 label='Validation F1', color='blue', linestyle='-')

        plt.title(f'Training & Validation F1 Score - {self.experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        dataset = self.config.get("dataset_name", "unknown")
        save_path = self.output_dir / f"{self.experiment_name}_{dataset}_f1_curves.png"

        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"F1 curves saved to {save_path}")

    def plot_confusion_matrix_torch(self, model, device, test_loader, label_dict):
        """
        Confusion matrix for PyTorch models.
        """
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                logits = outputs["out"]

                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > 0.5).astype(int)

                all_preds.extend(pred.flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='g',
            xticklabels=label_dict.values(),
            yticklabels=label_dict.values(),
            cmap='Blues'
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {self.experiment_name}")

        dataset = self.config.get("dataset_name", "unknown")

        save_path = self.output_dir / f"{dataset}_confusion_matrix.png"

        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {save_path}")

    def plot_f1_curve_torch(self, train_f1, val_f1):
        """
        Plot both Training and Validation F1 curves for PyTorch experiments.
        """
        if len(train_f1) == 0 and len(val_f1) == 0:
            print("[INFO] No F1 scores available to plot.")
            return

        plt.figure(figsize=(10, 5))

        # Plot training F1 if exists
        if len(train_f1) > 0:
            plt.plot(range(1, len(train_f1) + 1), train_f1,
                     label="Training F1", color="red", linewidth=2)

        # Plot validation F1 if exists
        if len(val_f1) > 0:
            plt.plot(range(1, len(val_f1) + 1), val_f1,
                     label="Validation F1", color="blue", linewidth=2)

        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.title(f"F1 Score Over Training - {self.experiment_name}")
        plt.grid(True)
        plt.legend()

        dataset = self.config.get("dataset_name", "unknown")

        save_path = self.output_dir / f"{self.experiment_name}_{dataset}_f1_curve.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[INFO] PyTorch F1 curves saved to: {save_path}")
