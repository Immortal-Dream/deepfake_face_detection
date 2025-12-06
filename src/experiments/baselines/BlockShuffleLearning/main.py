DATASET_NAME = "rvf10k"  # change each run

import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

print("PROJECT ROOT:", ROOT)

from configs.xception_bsl import *
from src.experiments.XceptionPyTorchExperiment import XceptionPyTorchExperiment
import argparse


from configs.rvf10k import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(DATASET_NAME)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = {
    "epochs": args.epochs,
    "batch_size": 32,
    "dataset_name": DATASET_NAME,   # <--- ADD THIS
}

print(f"Training for {args.epochs} epochs...")

# ------------------------------------------------
# ✅ 1. CREATE EXPERIMENT *BEFORE* TRAINING
# ------------------------------------------------
exp = XceptionPyTorchExperiment(config, model=net, device=device)
train.experiment = exp      # <-- link trainer → experiment
exp.f1_scores = []          # ensure list exists

# ------------------------------------------------
# ✅ 2. TRAINING LOOP (F1 SCORES NOW LOG CORRECTLY)
# ------------------------------------------------
for epoch in range(args.epochs):
    train.train(train_loader)
    train.val(val_loader)
    # if hasattr(exp, "should_stop") and exp.should_stop:
    #     print("\n✨ Training stopped because target metrics were reached!")
    #     break
#print("Training complete.")

# ------------------------------------------------
# ✅ 3. EVALUATE ON TEST SET
# ------------------------------------------------
metrics = exp.evaluate(test_loader)

# Save metrics & leaderboard
exp.save_metrics_to_csv()
exp.update_leaderboard()

print("FINAL METRICS:")
print(metrics)

# ------------------------------------------------
# ✅ 4. GENERATE PLOTS
# ------------------------------------------------
exp.plotter.plot_confusion_matrix_torch(
    model=exp.model,
    device=exp.device,
    test_loader=test_loader,
    label_dict={0: "real", 1: "fake"}
)

# F1 curve using logged F1 values
exp.plotter.plot_f1_curve_torch(
    train_f1=exp.train_f1_scores,
    val_f1=exp.val_f1_scores
)


print("Plots generated and saved.")
