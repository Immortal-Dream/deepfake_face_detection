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

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description="Train ViT baseline on a selected dataset")

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='rvf10k',
        help='Choose dataset: rvf10k, STARGAN, dalle2, midjourney, StableDiffusion, latent_diffusion, taming_transformer_VQGAN'
    )

    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='Path to save or load dataset CSV'
    )

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--test_split', type=float, default=0.3)
    parser.add_argument('--val_split', type=float, default=0.5)

    args = parser.parse_args()

    paths = get_dataset_paths(args.dataset_name)

    data_dirs = [str(paths["TRAIN_ROOT"]), str(paths["VALID_ROOT"])]

    # ------------------------------
    # Build configuration
    # ------------------------------
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "test_split": args.test_split,
        "val_split": args.val_split,
        "dataset_name": args.dataset_name,   
    }

    csv_path = OUTPUT_FOLDER / f"ViT_{args.dataset_name}.csv"

    if not csv_path.exists():
        print(f"Creating dataset CSV for {args.dataset_name} at {csv_path}")
        create_dataset_csv(data_dirs, csv_path)
    else:
        print(f"Using existing CSV for {args.dataset_name}: {csv_path}")


    print("\nLoading dataset...")
    X, y, label_dict = load_and_preprocess_data(
        csv_path, args.image_size, args.image_size
    )

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_split, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=args.val_split, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    vit_exp = ViTBaselineExperiment(config)
    vit_metrics = vit_exp.run(X_train, y_train, X_val, y_val, X_test, y_test, label_dict)

    print("\n" + "=" * 60)
    print(f"VIT BASELINE RESULTS on {args.dataset_name}")
    print("=" * 60)
    print(f"Accuracy   : {vit_metrics['accuracy']:.4f}")
    print(f"Precision  : {vit_metrics['precision']:.4f}")
    print(f"Recall     : {vit_metrics['recall']:.4f}")
    print(f"F1 Score   : {vit_metrics['f1_score']:.4f}")
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
