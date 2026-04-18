import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from dataset import BiscuitDataset, get_val_transform

DEVICE = torch.device("cpu")

NUM_CLASSES = 4
BATCH_SIZE = 16

TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "models/resnet50_biscuit.pth"

CLASS_NAMES = [
    "Defect_No",
    "Defect_Shape",
    "Defect_Object",
    "Defect_Color"
]

# ---------------- DATA ----------------
test_dataset = BiscuitDataset(
    TEST_CSV,
    transform=get_val_transform()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------- MODEL ----------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
