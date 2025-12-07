import sys, os
import argparse
import torch

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

# --------------------------------------------------
# IMPORT MODEL + TRAINER
# --------------------------------------------------
from configs.xception_bsl import net, train
from configs.rvf10k import build_dataloaders
from src.experiments.XceptionPyTorchExperiment import XceptionPyTorchExperiment

# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Cross-Dataset Training for Xception BSL")
parser.add_argument("--train_dataset", type=str, required=True)
parser.add_argument("--test_dataset", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)

args = parser.parse_args()

TRAIN_DATASET = args.train_dataset
TEST_DATASET = args.test_dataset

print(f"\n✅ TRAINING ON: {TRAIN_DATASET}")
print(f"✅ TESTING ON : {TEST_DATASET}")

# --------------------------------------------------
# LOAD DATALOADERS
# --------------------------------------------------
from torch.utils.data import ConcatDataset, DataLoader

# --------------------------------------------------
# MULTI-SOURCE TRAIN LOADER
# --------------------------------------------------
train_datasets = TRAIN_DATASET.split(",")   # supports: A,B,C
train_sets = []
val_sets = []

for d in train_datasets:
    tr, va, _ = build_dataloaders(d.strip())
    train_sets.append(tr.dataset)
    val_sets.append(va.dataset)

combined_train_set = ConcatDataset(train_sets)
combined_val_set   = ConcatDataset(val_sets)

train_loader = DataLoader(combined_train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(combined_val_set, batch_size=32, shuffle=False)

# --------------------------------------------------
# SINGLE-SOURCE TEST LOADER
# --------------------------------------------------
_, _, test_loader = build_dataloaders(TEST_DATASET)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
config = {
    "epochs": args.epochs,
    "batch_size": 32,
    "dataset_name": f"{TRAIN_DATASET}_to_{TEST_DATASET}",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# CREATE EXPERIMENT
# --------------------------------------------------
exp = XceptionPyTorchExperiment(config, model=net, device=device)
train.experiment = exp
exp.train_f1_scores = []
exp.val_f1_scores = []

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
for epoch in range(args.epochs):
    train.train(train_loader)
    train.val(val_loader)

print("\n✅ TRAINING DONE")

# --------------------------------------------------
# EVALUATE ON **OTHER DATASET**
# --------------------------------------------------
metrics = exp.evaluate(test_loader)
exp.save_metrics_to_csv()
exp.update_leaderboard()

# --------------------------------------------------
# PLOTS
# --------------------------------------------------
exp.plotter.plot_confusion_matrix_torch(
    model=exp.model,
    device=exp.device,
    test_loader=test_loader,
    label_dict={0: "real", 1: "fake"}
)

exp.plotter.plot_f1_curve_torch(
    train_f1=exp.train_f1_scores,
    val_f1=exp.val_f1_scores
)

print("\n✅ CROSS-DATASET EXPERIMENT COMPLETE")
print(metrics)
