import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.services.F1MetricsCallback import F1MetricsCallback


class PlotService:
    """
    Handles all visualization tasks for experiments.
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize plotter with experiment metadata.

        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for saving plots
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir

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

        save_path = self.output_dir / f'{self.experiment_name}_confusion_matrix.png'
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

        save_path = self.output_dir / f'{self.experiment_name}_f1_curves.png'
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"F1 curves saved to {save_path}")
