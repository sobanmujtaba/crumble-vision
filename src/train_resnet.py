# src/train_resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import (
    BiscuitDataset,
    get_train_transform,
    get_val_transform
)


# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"

MODEL_SAVE_PATH = "models/resnet50_biscuit.pth"


# ---------------- DATA ----------------
train_dataset = BiscuitDataset(
    TRAIN_CSV,
    transform=get_train_transform()
)

val_dataset = BiscuitDataset(
    VAL_CSV,
    transform=get_val_transform()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------- MODEL ----------------
model = models.resnet50(weights="IMAGENET1K_V1")

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------- TRAIN ----------------
def train_one_epoch():
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        acc = 100 * correct / total

        loop.set_description(f"Train Acc: {acc:.2f}%")

    return running_loss / len(train_loader), acc


# ---------------- VALIDATE ----------------
def validate():
    model.eval()

    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total

    return val_loss / len(val_loader), acc


# ---------------- MAIN ----------------
def main():
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Best model saved.")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
