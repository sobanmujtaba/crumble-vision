# src/dataset.py

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_MAP = {
    "Defect_No": 0,
    "Defect_Shape": 1,
    "Defect_Object": 2,
    "Defect_Color": 3
}


# ---------------- TRANSFORMS ----------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


# ---------------- DATASET ----------------
class BiscuitDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["full_path"]
        label_name = row["classDescription"]

        image = Image.open(image_path).convert("RGB")
        label = CLASS_MAP[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label
