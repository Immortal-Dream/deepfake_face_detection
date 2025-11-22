import argparse
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config.path_config import TRAIN_ROOT, VALID_ROOT, OUTPUT_FOLDER
from src.utils.csv_utils import create_dataset_csv
from src.utils.data_loader_utils import load_and_preprocess_data
from src.experiments.baselines.ViT.ViTBaselineExperiment import ViTBaselineExperiment

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description='Train ViT-Base-Patch16-224 for Fake vs Real Face Classification'
    )
    parser.add_argument(
        '--data_dirs', nargs='+',
        default=[str(TRAIN_ROOT), str(VALID_ROOT)],
        help='List of directories containing class subfolders'
    )
    parser.add_argument(
        '--csv_path',
        default=str(OUTPUT_FOLDER / "ViT_dataset.csv"),
        help='Path to dataset CSV file'
    )
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (height, width)')
    parser.add_argument('--test_split', type=float, default=0.3, help='Test set ratio')
    parser.add_argument('--val_split', type=float, default=0.5, help='Validation ratio within temp set')

    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'test_split': args.test_split,
        'val_split': args.val_split,
    }

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        create_dataset_csv(args.data_dirs, csv_path)
    else:
        print(f"Using existing CSV file: {csv_path}")

    print("\nLoading dataset...")
    X, y, label_dict = load_and_preprocess_data(csv_path, args.image_size, args.image_size)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=args.test_split, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=args.val_split, random_state=42)

    print(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    vit_exp = ViTBaselineExperiment(config)
    vit_metrics = vit_exp.run(X_train, y_train, X_val, y_val, X_test, y_test, label_dict)

    print("\n" + "=" * 60)
    print("VIT BASELINE SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy: {vit_metrics['accuracy']:.4f}")
    print(f"Test Precision: {vit_metrics['precision']:.4f}")
    print(f"Test Recall: {vit_metrics['recall']:.4f}")
    print(f"Test F1 Score: {vit_metrics['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main()
