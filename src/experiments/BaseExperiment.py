import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.config.path_config import OUTPUT_FOLDER, MODEL_FOLDER, OVERALL_FOLDER
from src.services.F1MetricsCallback import F1MetricsCallback
from src.services.plot_service import PlotService

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
        self.plotter = PlotService(experiment_name, self.output_dir)
        self._setup_directories()

    # $Requires$ customization according to the model structure of the specific method.
    def create_model(self, num_classes: int):
        """
        Create and compile the model. Must be implemented by subclasses.

        Args:
            num_classes: Number of output classes

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement create_model method")

    # $Requires$ customization according to the model training process of the specific method.
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

    # $Could$ be customized according to specific pipeline.
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

        # Plot results using the decoupled plotter
        self.plotter.plot_f1_curves(f1_callback)
        self.plotter.plot_confusion_matrix(self.model, X_test, y_test, label_dict)

        # Save results
        self.save_metrics_to_csv()
        self.update_leaderboard()
        self.save_model()

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETED: {self.experiment_name}")
        print(f"{'=' * 60}\n")

        return metrics

    # This method is universal for all Keras models
    def save_model(self):
        """Save the trained model."""
        model_path = MODEL_FOLDER / f'{self.experiment_name}_model.h5'
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

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

    def update_leaderboard(self):
        """
        Update or insert experiment metrics in the leaderboard CSV.
        Sorts by f1_score in descending order.
        """
        # Check for required metrics
        required_keys = ['experiment_name', 'precision', 'recall', 'f1_score', 'timestamp']
        if not all(key in self.metrics for key in required_keys):
            print("Warning: Missing required metrics for leaderboard. Skipping update.")
            return

        leaderboard_path = OVERALL_FOLDER / 'leader_board.csv'

        # Ensure directory exists
        OVERALL_FOLDER.mkdir(parents=True, exist_ok=True)

        # Prepare leaderboard data (only needed columns)
        leaderboard_data = {
            'experiment_name': self.metrics['experiment_name'],
            'precision': f"{self.metrics['precision']:.4f}",
            'recall': f"{self.metrics['recall']:.4f}",
            'f1_score': f"{self.metrics['f1_score']:.4f}",
            'timestamp': self.metrics['timestamp']
        }

        # Read existing leaderboard or create new
        if leaderboard_path.exists():
            df = pd.read_csv(leaderboard_path)

            # Update existing or append new
            if self.experiment_name in df['experiment_name'].values:
                df.loc[df['experiment_name'] == self.experiment_name] = list(leaderboard_data.values())
                print(f"Updated existing entry for {self.experiment_name} in leaderboard")
            else:
                df = pd.concat([df, pd.DataFrame([leaderboard_data])], ignore_index=True)
                print(f"Added new entry for {self.experiment_name} to leaderboard")
        else:
            df = pd.DataFrame([leaderboard_data])
            print(f"Created new leaderboard with entry for {self.experiment_name}")

        # Sort by f1_score descending and save
        df['f1_score'] = df['f1_score'].astype(float)  # Ensure numeric for sorting
        df = df.sort_values(by='f1_score', ascending=False).reset_index(drop=True)
        df.to_csv(leaderboard_path, index=False)
        print(f"Leaderboard sorted and saved to {leaderboard_path}")

    def _setup_directories(self):
        """Create experiment output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment output directory: {self.output_dir}")