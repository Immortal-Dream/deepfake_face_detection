import argparse
import os
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config.path_config import *
from src.utils.csv_utils import create_dataset_csv
from src.utils.data_loader_utils import load_and_preprocess_data
from src.utils.model_utils import F1MetricsCallback
from src.experiments.baselines.MobileNetV3Large.MobileNetV3LargeBaselineExperiment import MobileNetV3LargeBaselineExperiment
from src.experiments.baselines.MobileNetV3Large.MobileNetV3LargeAugumentedExperiment import MobileNetV3LargeAugumentedExperiment

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """
    Main execution function that runs experiments.
    """
    parser = argparse.ArgumentParser(
        description='Train MobileNetV3Large for Fake vs Real Face Classification'
    )
    parser.add_argument('--data_dirs', nargs='+',
                        default=[str(TRAIN_ROOT), str(VALID_ROOT)],
                        help='List of directories containing class subfolders')
    parser.add_argument('--csv_path',
                        default=str(OUTPUT_FOLDER / "MobileNetV3Large_dataset.csv"),
                        help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (height, width)')
    parser.add_argument('--run_baseline', action='store_true', default=True,
                        help='Run baseline experiment')
    parser.add_argument('--run_augmented', action='store_true', default=True,
                        help='Run augmented experiment')
    parser.add_argument('--no_baseline', dest='run_baseline', action='store_false',
                        help='Skip baseline experiment')
    parser.add_argument('--no_augmented', dest='run_augmented', action='store_false',
                        help='Skip augmented experiment')
    parser.add_argument('--test_split', type=float, default=0.3,
                        help='Test set ratio')
    parser.add_argument('--val_split', type=float, default=0.5,
                        help='Validation ratio from test set')

    args = parser.parse_args()

    # Configuration dictionary
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'test_split': args.test_split,
        'val_split': args.val_split,
    }

    # Create or load dataset CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        create_dataset_csv(args.data_dirs, csv_path)
    else:
        print(f"Using existing CSV file: {csv_path}")

    # Load data (only once for both experiments)
    print("\nLoading dataset...")
    X, y, label_dict = load_and_preprocess_data(csv_path, args.image_size, args.image_size)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_split, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=args.val_split, random_state=42
    )

    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Ensure output directories exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    # Run experiments
    baseline_metrics = None
    augmented_metrics = None

    if args.run_baseline:
        baseline_exp = MobileNetV3LargeBaselineExperiment(config)
        baseline_metrics = baseline_exp.run(X_train, y_train, X_val, y_val, X_test, y_test, label_dict)

    if args.run_augmented:
        augmented_exp = MobileNetV3LargeAugumentedExperiment(config)
        augmented_metrics = augmented_exp.run(X_train, y_train, X_val, y_val, X_test, y_test, label_dict)

    # Summary comparison
    if baseline_metrics and augmented_metrics:
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Baseline Test Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"Augmented Test Accuracy: {augmented_metrics['accuracy']:.4f}")
        print(f"Baseline Test F1 Score: {baseline_metrics['f1_score']:.4f}")
        print(f"Augmented Test F1 Score: {augmented_metrics['f1_score']:.4f}")
        print(f"Improvement (Accuracy): {augmented_metrics['accuracy'] - baseline_metrics['accuracy']:.4f}")
        print(f"Improvement (F1): {augmented_metrics['f1_score'] - baseline_metrics['f1_score']:.4f}")
        print("="*60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)

    # Run main
    main()