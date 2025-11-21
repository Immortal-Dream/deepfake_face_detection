import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

print("PROJECT ROOT:", ROOT)


from configs.xception_bsl import *
from src.experiments.XceptionPyTorchExperiment import XceptionPyTorchExperiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

config = {
    "epochs": args.epochs,
    "batch_size": 32
}

print(f"Training for {args.epochs} epochs...")

# -----------------------------
# TRAINING LOOP (your original)
# -----------------------------
for epoch in range(args.epochs):
    train.train(train_loader)
    train.val(val_loader)

print("Training complete.")

# -----------------------------
# EVALUATE WITH EXPERIMENT LOGGING
# -----------------------------
exp = XceptionPyTorchExperiment(config, model=net, device=device)
metrics = exp.evaluate(test_loader)
exp.save_metrics_to_csv()
exp.update_leaderboard()

print("FINAL METRICS:")
print(metrics)
