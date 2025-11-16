import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from src.config.path_config import OUTPUT_FOLDER, MODEL_FOLDER
from src.utils.model_utils import F1MetricsCallback

warnings.filterwarnings('ignore')


class BaseExperiment:
    """
    Base abstract class that defines the structure for an experiment.
    Subclasses must implement create_model() and train() methods.
    """

    def __init__(self, experiment_name: str, config: dict):
        """
        Initialize experiment with name and configuration.

        Args:
            experiment_name: Name of the experiment
            config: Dictionary containing hyperparameters and settings
        """
        self.experiment_name = experiment_name
        self.config = config
        self.model = None
        self.history = None
        self.metrics = {}
        self.output_dir = OUTPUT_FOLDER / self.experiment_name
        self._setup_directories()

    def create_model(self, num_classes: int):
        """
        Create and compile the model. Must be implemented by subclasses.

        Args:
            num_classes: Number of output classes

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement create_model method")

    def train(self, X_train, y_train, X_val, y_val, f1_callback: F1MetricsCallback):
        """
        Main training loop for the experiment. Must be implemented by subclasses.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            f1_callback: Callback instance for tracking F1 scores

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement train method")

    def _setup_directories(self):
        """Create experiment output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment output directory: {self.output_dir}")

    def evaluate(self, X_test, y_test, label_dict: dict):
        """
        Evaluate model and compute metrics.

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating {self.experiment_name}...")

        y_pred = self.model.predict(X_test, verbose=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        accuracy = accuracy_score(y_true_labels, y_pred_labels)

        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': self.experiment_name,
        }
        # Add hyperparameters to metrics
        self.metrics.update(self.config)

        return self.metrics

    def plot_confusion_matrix(self, X_test, y_test, label_dict: dict):
        """Generate and save confusion matrix plot."""
        y_pred = self.model.predict(X_test, verbose=0)
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
        """Plot training and validation F1 score curves."""
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

    def save_metrics_to_csv(self):
        """Save metrics and configuration to CSV log file."""
        metrics_df = pd.DataFrame([self.metrics])
        csv_path = self.output_dir / 'metrics_log.csv'

        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
        else:
            metrics_df.to_csv(csv_path, index=False)

        print(f"Metrics log saved to {csv_path}")

    def save_model(self):
        """Save the trained model."""
        model_path = MODEL_FOLDER / f'{self.experiment_name}_model.h5'
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test, label_dict: dict):
        """
        Execute the complete experiment workflow.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            label_dict: Label mapping dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"STARTING EXPERIMENT: {self.experiment_name}")
        print(f"{'=' * 60}")

        # Create model
        self.create_model(len(label_dict))

        # Create F1 callback
        f1_callback = F1MetricsCallback((X_train, y_train), (X_val, y_val))

        # Train model
        self.history = self.train(X_train, y_train, X_val, y_val, f1_callback)

        # Evaluate model
        metrics = self.evaluate(X_test, y_test, label_dict)
        print("\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Plot results
        self.plot_f1_curves(f1_callback)
        self.plot_confusion_matrix(X_test, y_test, label_dict)

        # Save results
        self.save_metrics_to_csv()
        self.save_model()

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        print(f"{'=' * 60}\n")

        return metrics