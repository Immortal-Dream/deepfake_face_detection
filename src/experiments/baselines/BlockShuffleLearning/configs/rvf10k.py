import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from config.path_config import get_dataset_paths
from src.config.path_config import get_dataset_paths


class RVF10KDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_raw = int(row["label"])
        label = 1 - label_raw

        folder = "real" if label_raw == 1 else "fake"
        img_path = os.path.join(self.img_root, folder, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([label], dtype=torch.float32)


def build_dataloaders(dataset_name):

    paths = get_dataset_paths(dataset_name)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_set = RVF10KDataset(paths["TRAIN_CSV"], paths["TRAIN_ROOT"], transform)
    val_set   = RVF10KDataset(paths["VALID_CSV"], paths["VALID_ROOT"], transform)

    return (
        DataLoader(train_set, batch_size=8, shuffle=True),
        DataLoader(val_set, batch_size=8, shuffle=False),
        DataLoader(val_set, batch_size=8, shuffle=False)
    )
