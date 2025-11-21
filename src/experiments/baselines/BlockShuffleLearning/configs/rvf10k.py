import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------
#  Custom Dataset
# --------------------------
class RVF10KDataset(Dataset):
    def __init__(self, csv_path, img_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # RVF10K: 1 = real, 0 = fake
        label_raw = int(row["label"])  

        # Xception-BSL expects: 0 = real, 1 = fake
        label = 1 - label_raw            # flip labels

        # Choose folder based on ORIGINAL label
        label_folder = "real" if label_raw == 1 else "fake"

        # Build full image path
        img_path = os.path.join(self.img_folder, label_folder, row["filename"])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Model expects shape [batch, 1]
        label = torch.tensor([label], dtype=torch.float32)

        return image, label



# --------------------------
#  Paths
# --------------------------
# root = "./datasets/RVF10K"   # <-- adjust to your actual path

# train_csv = os.path.join(root, "train.csv")
# valid_csv = os.path.join(root, "valid.csv")

# train_img_folder = os.path.join(root, "train")
# valid_img_folder = os.path.join(root, "valid")

import sys
from pathlib import Path

# ---------------------------------------------------------
# AUTO-DETECT THE PATH TO <project>/src WHERE config/ LIVES
# ---------------------------------------------------------
CURRENT = Path(__file__).resolve()

# Walk upward until config/path_config.py is found
SRC_DIR = None
for parent in CURRENT.parents:
    candidate = parent / "config" / "path_config.py"
    if candidate.exists():
        SRC_DIR = parent
        break

if SRC_DIR is None:
    raise RuntimeError("Could not locate path_config.py in parent directories.")

# Add the src directory to sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.path_config import (
    TRAIN_CSV, VALID_CSV,
    TRAIN_ROOT, VALID_ROOT
)


# --------------------------
#  Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# --------------------------
#  Dataset Instances
# --------------------------
# train_set = RVF10KDataset(train_csv, train_img_folder, transform)
# val_set   = RVF10KDataset(valid_csv, valid_img_folder, transform)

train_set = RVF10KDataset(TRAIN_CSV, TRAIN_ROOT, transform)
val_set   = RVF10KDataset(VALID_CSV, VALID_ROOT, transform)


# --------------------------
#  DataLoaders
# --------------------------
dataloader_train = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
dataloader_val   = DataLoader(val_set,   batch_size=8, shuffle=False, num_workers=0)
dataloader_test  = dataloader_val   # RVF10K does not have a separate test set
