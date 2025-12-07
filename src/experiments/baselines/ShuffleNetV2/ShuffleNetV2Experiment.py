import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from experiments.BaseExperiment import BaseExperiment
from src.config.path_config import OUTPUT_FOLDER, OVERALL_FOLDER, MODEL_FOLDER


class ShuffleNetV2Experiment(BaseExperiment):
    """
    PyTorch baseline experiment for ShuffleNetV2 (1.0x) following the same
    output/logging conventions as other baselines (e.g., MobileNetV3Large).
    """

    def __init__(self, config: dict):
        self.experiment_name = "ShuffleNetV2_pytorch"
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = {"train_f1": [], "val_f1": []}
        self.metrics = {}
        self.output_dir = OUTPUT_FOLDER / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment output directory: {self.output_dir}")

    def create_model(self, num_classes: int):
        from torchvision.models import shufflenet_v2_x1_0
        model = shufflenet_v2_x1_0(weights="DEFAULT")
        in_features = model.fc.in_features
        # Binary output (logit)
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model.to(self.device)
        return self.model

    def _forward_batch(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)
        return outputs, labels

    def train(self, train_loader, val_loader):
        epochs = self.config["epochs"]
        lr = self.config.get("lr", 1e-3)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"Device: {self.device} | LR: {lr} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_preds = []
            epoch_trues = []
            running_loss = 0.0
            start_epoch = time.time()
            batch_times = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                t0 = time.time()
                optimizer.zero_grad()
                outputs, labels = self._forward_batch(batch)
                # labels shape [batch,1]; outputs shape [batch,1]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int).flatten()
                epoch_preds.extend(preds)
                epoch_trues.extend(labels.detach().cpu().numpy().flatten())
                batch_times.append(time.time() - t0)
            avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0.0
            train_f1 = f1_score(epoch_trues, epoch_preds, average='weighted')
            self.history['train_f1'].append(train_f1)

            # Validation
            self.model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs, labels = self._forward_batch(batch)
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds = (probs > 0.5).astype(int).flatten()
                    val_preds.extend(preds)
                    val_trues.extend(labels.detach().cpu().numpy().flatten())
            val_f1 = f1_score(val_trues, val_preds, average='weighted')
            self.history['val_f1'].append(val_f1)
            epoch_time = time.time() - start_epoch
            print(
                f"Epoch {epoch}/{epochs} | time: {epoch_time:.1f}s | avg_batch: {avg_batch:.3f}s | train_f1: {train_f1:.4f} | val_f1: {val_f1:.4f}")
        return self.history

    def evaluate(self, test_loader, label_dict: dict):
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in test_loader:
                outputs, labels = self._forward_batch(batch)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                pred = (probs > 0.5).astype(int).flatten()
                preds.extend(pred)
                trues.extend(labels.detach().cpu().numpy().flatten())
        precision = precision_score(trues, preds, average='weighted', zero_division=0)
        recall = recall_score(trues, preds, average='weighted', zero_division=0)
        f1 = f1_score(trues, preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(trues, preds)
        self.metrics = {
            'experiment_name': self.experiment_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.metrics.update(self.config)
        print("Final Metrics:")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        # Plot artifacts
        self._plot_f1_curves()
        self._plot_confusion_matrix(trues, preds, label_dict)
        self.save_metrics_to_csv()
        self.update_leaderboard()
        self.save_model()
        return self.metrics

    def save_model(self):
        MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
        path = MODEL_FOLDER / "ShuffleNetV2_pytorch.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def save_metrics_to_csv(self):
        csv_path = self.output_dir / 'metrics_log.csv'
        df_new = pd.DataFrame([self.metrics])
        if csv_path.exists():
            df_exist = pd.read_csv(csv_path)
            df_all = pd.concat([df_exist, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(csv_path, index=False)
        print(f"Metrics log saved to {csv_path}")

    def update_leaderboard(self):
        OVERALL_FOLDER.mkdir(parents=True, exist_ok=True)
        leaderboard_path = OVERALL_FOLDER / 'leader_board.csv'
        row = {
            'experiment_name': self.metrics['experiment_name'],
            'precision': f"{self.metrics['precision']:.4f}",
            'recall': f"{self.metrics['recall']:.4f}",
            'f1_score': f"{self.metrics['f1_score']:.4f}",
            'timestamp': self.metrics['timestamp']
        }
        new_row_df = pd.DataFrame([row])
        if leaderboard_path.exists():
            df = pd.read_csv(leaderboard_path)
            if self.experiment_name in df['experiment_name'].values:
                df.loc[df['experiment_name'] == self.experiment_name] = list(row.values())
            else:
                df = pd.concat([df, new_row_df], ignore_index=True)
        else:
            df = new_row_df
        df['f1_score'] = df['f1_score'].astype(float)
        df = df.sort_values(by='f1_score', ascending=False).reset_index(drop=True)
        df.to_csv(leaderboard_path, index=False)
        print(f"Leaderboard updated → {leaderboard_path}")

    def _plot_f1_curves(self):
        if not self.history['train_f1']:
            print("No F1 history to plot.")
            return
        epochs = range(1, len(self.history['train_f1']) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_f1'], label='Training F1', color='red')
        plt.plot(epochs, self.history['val_f1'], label='Validation F1', color='blue')
        plt.title(f'Training & Validation F1 Score - {self.experiment_name}')
        plt.xlabel('Epoch');
        plt.ylabel('F1 Score');
        plt.legend();
        plt.grid(True)
        save_path = self.output_dir / f'{self.experiment_name}_f1_curves.png'
        plt.savefig(save_path, dpi=300);
        plt.close()
        print(f"F1 curves saved to {save_path}")

    def _plot_confusion_matrix(self, y_true, y_pred, label_dict: dict):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
        plt.xlabel('Predicted');
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.experiment_name}')
        plt.tight_layout()
        save_path = self.output_dir / f'{self.experiment_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300);
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
