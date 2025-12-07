import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from src.config.path_config import OVERALL_FOLDER
import pandas as pd

from src.experiments.BaseExperiment import BaseExperiment
from src.config.path_config import MODEL_FOLDER


class ViTBaselineExperiment(BaseExperiment):
    def __init__(self, config: dict):
        dataset = config.get("dataset_name", "rvf10k")
        exp_name = "ViT_baseline"
        super().__init__(exp_name, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = None

    def create_model(self, num_classes: int):
        base_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        base_model.classifier = nn.Linear(base_model.config.hidden_size, num_classes)
        self.vit_model = base_model.to(self.device)
        self.model = ViTWrapper(self.vit_model, self.processor, self.device, self.config)

    def train(self, X_train, y_train, X_val, y_val, f1_callback):
        print("\n" + "=" * 60)
        print("TRAINING VIT BASELINE")
        print("=" * 60)

        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_loader(X_val, y_val, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.vit_model.parameters(), lr=self.config.get("learning_rate", 3e-4))

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            train_metrics = self._run_epoch(train_loader, criterion, optimizer, training=True)
            val_metrics = self._run_epoch(val_loader, criterion, optimizer=None, training=False)

            history["loss"].append(train_metrics["loss"])
            history["accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            if f1_callback is not None:
                logs = {
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
                f1_callback.model = self.model
                f1_callback.on_epoch_end(epoch, logs)

            print(
                f"loss: {train_metrics['loss']:.4f} - accuracy: {train_metrics['accuracy']:.4f} - "
                f"val_loss: {val_metrics['loss']:.4f} - val_accuracy: {val_metrics['accuracy']:.4f}"
            )

        return history

    def _create_loader(self, X, y, shuffle):
        dataset = NumpyToViTDataset(X, y, self.processor)
        use_cuda = self.device.type == "cuda"
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            num_workers=4 if use_cuda else 0,
            pin_memory=use_cuda,
        )

    def _run_epoch(self, data_loader, criterion, optimizer=None, training=True):
        if training:
            self.vit_model.train()
        else:
            self.vit_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(
            data_loader,
            desc="Train" if training else "Validation",
            leave=False,
            ncols=100
        )

        for images, labels in iterator:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if training:
                optimizer.zero_grad()

            outputs = self.vit_model(pixel_values=images).logits
            loss = criterion(outputs, labels.long())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            iterator.set_postfix({
                "loss": f"{total_loss / max(total, 1):.4f}",
                "acc": f"{(correct / max(total, 1)) * 100:.2f}%"
            })

        return {"loss": total_loss / total, "accuracy": correct / total}

    def save_model(self):
        dataset = self.config.get("dataset_name", "rvf10k")
        dataset_suffix = f"_{dataset}" if dataset != "rvf10k" else ""
        model_path = MODEL_FOLDER / f"{self.experiment_name}{dataset_suffix}_model.pth"
        torch.save(
            {
                "model_state_dict": self.vit_model.state_dict(),
                "config": self.config,
                "processor": self.processor,
            },
            model_path,
        )
        print(f"Model saved to {model_path}")
    
    

    def update_leaderboard(self):
        """Dataset-specific leen insert the update_leaderboard() method inside the class.
        
        aderboard like Xception."""
        OVERALL_FOLDER.mkdir(parents=True, exist_ok=True)

        dataset = self.config.get("dataset_name", "rvf10k")
        leaderboard_path = OVERALL_FOLDER / f"{dataset}_leaderboard.csv"

        row = {
            "experiment_name": self.experiment_name,
            "precision": self.metrics["precision"],
            "recall": self.metrics["recall"],
            "f1_score": self.metrics["f1_score"],
            "accuracy": self.metrics["accuracy"],
            "timestamp": self.metrics["timestamp"],
        }

        new_row_df = pd.DataFrame([row])

        if leaderboard_path.exists():
            df = pd.read_csv(leaderboard_path)

            if self.experiment_name in df["experiment_name"].values:
                for col in row:
                    df.loc[df["experiment_name"] == self.experiment_name, col] = row[col]
                print(f"[INFO] Updated {self.experiment_name} in leaderboard")
            else:
                df = pd.concat([df, new_row_df], ignore_index=True)
                print(f"[INFO] Added new entry for {self.experiment_name} to leaderboard")
        else:
            df = new_row_df
            print(f"[INFO] Created new leaderboard file for {dataset}")

        df = df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        df.to_csv(leaderboard_path, index=False)

        print(f"[INFO] Saved leaderboard → {leaderboard_path}")



class ViTWrapper:
    """Wrapper for BaseExperiment compatibility."""

    def __init__(self, vit_model, processor, device, config):
        self.vit_model = vit_model
        self.processor = processor
        self.device = device
        self.config = config

    def predict(self, X_test, verbose=1):
        self.vit_model.eval()
        batch_size = self.config.get("batch_size", 16)
        predictions = []

        if verbose:
            print(f"Predicting on {len(X_test)} samples...")

        for idx in range(0, len(X_test), batch_size):
            batch = X_test[idx : idx + batch_size]
            batch_preds = []

            for img in batch:
                image_array = img.astype(np.uint8)
                image = Image.fromarray(image_array)
                inputs = self.processor(image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.no_grad():
                    outputs = self.vit_model(pixel_values=pixel_values).logits
                    probs = torch.softmax(outputs, dim=1)
                    batch_preds.append(probs.cpu().numpy())

            predictions.extend(batch_preds)

            if verbose and idx % 100 == 0:
                print(f"Processed {min(idx + batch_size, len(X_test))}/{len(X_test)} samples")

        return np.vstack(predictions)


class NumpyToViTDataset(Dataset):
    """Dataset wrapper that converts numpy arrays into ViT inputs."""

    def __init__(self, X, y, processor):
        self.X = X
        self.y = self._prepare_labels(y)
        self.processor = processor

    def _prepare_labels(self, labels):
        if isinstance(labels, np.ndarray) and labels.ndim > 1:
            return np.argmax(labels, axis=1).astype(np.int64)
        return np.array(labels).astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_array = self.X[idx].astype(np.uint8)
        image = Image.fromarray(image_array)
        inputs = self.processor(image, return_tensors="pt")
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return inputs["pixel_values"].squeeze(0), label
