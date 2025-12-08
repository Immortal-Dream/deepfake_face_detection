import argparse
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config.path_config import get_dataset_paths, OUTPUT_FOLDER
from src.utils.csv_utils import create_dataset_csv
from src.utils.data_loader_utils import load_and_preprocess_data
from src.experiments.baselines.ViT.ViTBaselineExperiment import ViTBaselineExperiment

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="ViT Cross-Dataset Experiment")

    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Comma-separated datasets for training")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Dataset for testing")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.5)

    args = parser.parse_args()

    train_datasets = args.train_dataset.split(",")
    test_dataset = args.test_dataset

    print(f"\n✅ TRAINING ON: {train_datasets}")
    print(f"✅ TESTING ON : {test_dataset}")

    # -------------------------------
    # Build train directories
    # -------------------------------
    train_dirs = []
    for name in train_datasets:
        paths = get_dataset_paths(name)
        train_dirs.append(str(paths["TRAIN_ROOT"]))
        train_dirs.append(str(paths["VALID_ROOT"]))

    # -------------------------------
    # Build test directories
    # -------------------------------
    test_paths = get_dataset_paths(test_dataset)
    test_dirs = [str(test_paths["TRAIN_ROOT"]), str(test_paths["VALID_ROOT"])]

    # -------------------------------
    # CSV paths
    # -------------------------------
    train_csv = OUTPUT_FOLDER / f"ViT_TRAIN_{','.join(train_datasets)}.csv"

    test_csv = OUTPUT_FOLDER / f"ViT_TEST_{test_dataset}.csv"

    if not train_csv.exists():
        print(f"Creating training CSV → {train_csv}")
        create_dataset_csv(train_dirs, train_csv)

    if not test_csv.exists():
        print(f"Creating test CSV → {test_csv}")
        create_dataset_csv(test_dirs, test_csv)

    # -------------------------------
    # Load datasets
    # -------------------------------
    print("\n📥 Loading training data...")
    X_train_full, y_train_full, label_dict = load_and_preprocess_data(
        train_csv, args.image_size, args.image_size)

    print("\n📥 Loading test data...")
    X_test, y_test, _ = load_and_preprocess_data(
        test_csv, args.image_size, args.image_size)

    # -------------------------------
    # Split TRAIN → train / val
    # -------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_split,
        random_state=42
    )

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -------------------------------
    # Config
    # -------------------------------
    dataset_tag = f"{','.join(train_datasets)}_to_{test_dataset}"


    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "dataset_name": dataset_tag,
    }

    # -------------------------------
    # Run Experiment
    # -------------------------------
    vit_exp = ViTBaselineExperiment(config)
    vit_metrics = vit_exp.run(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_dict
    )

    print("\n✅ CROSS-DATASET EXPERIMENT COMPLETE")
    print(vit_metrics)


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
