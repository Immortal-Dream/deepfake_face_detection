import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.config.path_config import TRAIN_CSV, VALID_CSV, TRAIN_ROOT, VALID_ROOT
from src.experiments.baselines.ShuffleNetV2.ShuffleNetV2Experiment import ShuffleNetV2Experiment
from src.utils.csv_utils import create_dataset_csv
from src.experiments.baselines.BlockShuffleLearning.configs import rvf10k  # reuse existing dataset definitions


def main():
    parser = argparse.ArgumentParser(description='Train ShuffleNetV2 baseline on RVF10K dataset')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32, help='Override dataloader batch size (rvf10k default=8)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=224, help='(Reserved) Image size (rvf10k fixed 224)')
    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'image_size': args.image_size
    }

    # Ensure dataset CSVs exist (train/valid). If missing, create minimal ones.
    if not TRAIN_CSV.exists() or not VALID_CSV.exists():
        print('Train/Valid CSV not found. Creating from folder structure...')
        create_dataset_csv([str(TRAIN_ROOT)], TRAIN_CSV)
        create_dataset_csv([str(VALID_ROOT)], VALID_CSV)
    else:
        print(f'Using existing CSVs: {TRAIN_CSV}, {VALID_CSV}')

    # Always (re)build dataloaders with requested batch size (cannot mutate existing DataLoader)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(rvf10k.train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(rvf10k.val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    # RVF10K has no separate test set; use validation as test
    test_loader = val_loader
    print(f"DataLoaders built | batch_size={args.batch_size} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")
    label_dict = {'real': 0, 'fake': 1}  # rvf10k after flip: 0=real,1=fake

    exp = ShuffleNetV2Experiment(config)
    exp.create_model(num_classes=1)

    # Train
    print("\n======== TRAINING START ========")
    try:
        exp.train(train_loader, val_loader)
        print("======== TRAINING END ========\n")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user. Proceeding to evaluation...\n")

    # Evaluate + logging
    metrics = exp.evaluate(test_loader, label_dict)
    print('\nFINAL RESULTS:')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    main()
