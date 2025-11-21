import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.experiments.BaseExperiment import BaseExperiment

class XceptionPyTorchExperiment(BaseExperiment):
    """
    PyTorch version of BaseExperiment for the XceptionBSL model.
    """

    def __init__(self, config, model, device):
        super().__init__("Xception_BSL_baseline", config)
        self.model = model
        self.device = device

    def create_model(self, num_classes):
        """Already created in configs.xception_bsl, so skip."""
        return self.model

    def train(self, train_loader, val_loader):
        """
        Use your existing training loop (train.train(...)).
        """
        print("PyTorch training loop handled externally.")
        # nothing to do here because training happens in main.py
        return None

    def evaluate(self, test_loader):
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)

                # The model returns a dict with keys like "bsl", "bs", "rs"
                outputs = self.model(X)

                # print("DEBUG OUTPUT TYPE:", type(outputs))
                # print("DEBUG OUTPUT CONTENT:", outputs)
                # print("DEBUG OUTPUT KEYS (if dict):", getattr(outputs, "keys", lambda: None)())

                # raise SystemExit("STOP AFTER DEBUGGING")

                # Extract the main classification logits (your BCE loss comes from this)
                logits = outputs["out"]     # <-- main classifier output

                # Convert logits → probabilities using sigmoid
                prob = torch.sigmoid(logits).cpu().numpy()

                # Convert probabilities to binary predictions
                pred = (prob > 0.5).astype(int)

                preds.extend(pred.flatten())
                trues.extend(y.cpu().numpy().flatten())

        # Compute metrics
        precision = precision_score(trues, preds, zero_division=0)
        recall = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds, zero_division=0)
        accuracy = accuracy_score(trues, preds)

        # Save metrics
        self.metrics = {
            "experiment_name": self.experiment_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Include config (epochs, batch size, etc.)
        self.metrics.update(self.config)

        return self.metrics


        def update_leaderboard(self):
            """Update or append experiment metrics in the leaderboard safely."""

            OVERALL_FOLDER.mkdir(parents=True, exist_ok=True)
            leaderboard_path = OVERALL_FOLDER / "leader_board.csv"

            # Row for this experiment
            row = {
                "experiment_name": self.experiment_name,
                "precision": self.metrics["precision"],
                "recall": self.metrics["recall"],
                "f1_score": self.metrics["f1_score"],
                "accuracy": self.metrics["accuracy"],
                "timestamp": self.metrics["timestamp"],
            }

            new_row_df = pd.DataFrame([row])

            # Case 1: Leaderboard exists -> update OR append
            if leaderboard_path.exists():
                df = pd.read_csv(leaderboard_path)

                if self.experiment_name in df["experiment_name"].values:
                    # overwrite ONLY your row
                    df.loc[df["experiment_name"] == self.experiment_name] = row
                    print(f"[INFO] Updated existing entry for {self.experiment_name}")
                else:
                    # append new row
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    print(f"[INFO] Added new entry for {self.experiment_name}")

            # Case 2: Leaderboard does not exist -> create new file
            else:
                df = new_row_df
                print(f"[INFO] Created new leaderboard file")

            # Sort by f1 score descending
            df = df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

            # Save
            df.to_csv(leaderboard_path, index=False)
            print(f"[INFO] Leaderboard updated → {leaderboard_path}")
