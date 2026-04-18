import cv2
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torchvision import models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_val_transform

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")

MODEL_PATH = "models/resnet50_biscuit.pth"
TRAY_DIR = Path("assets/trays")
MANIFEST_PATH = TRAY_DIR / "tray_manifest.csv"
OUTPUT_PATH = TRAY_DIR / "tray_predictions.csv"

CELL_SIZE = 224
PADDING = 20

CLASS_NAMES = [
    "Defect_No",
    "Defect_Shape",
    "Defect_Object",
    "Defect_Color"
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ---------------- LOAD MODEL ----------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

transform = get_val_transform()


# ---------------- INFERENCE ----------------
def predict_crop(crop):
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return CLASS_NAMES[pred]


# ---------------- MAIN ----------------
def run_all_trays():
    manifest = pd.read_csv(MANIFEST_PATH)

    results = []

    for tray_id, tray_rows in manifest.groupby("tray_id"):
        tray_path = TRAY_DIR / f"{tray_id}.jpg"

        img = cv2.imread(str(tray_path))

        good_count = 0
        total_count = 0

        print(f"\nProcessing {tray_id}")

        for _, row in tray_rows.iterrows():
            x = int(row["x"])
            y = int(row["y"])

            crop = img[y:y+CELL_SIZE, x:x+CELL_SIZE]

            pred_label = predict_crop(crop)
            true_label = row["true_label"]

            is_correct = pred_label == true_label

            if pred_label == "Defect_No":
                good_count += 1

            total_count += 1

            results.append({
                "tray_id": tray_id,
                "row": row["row"],
                "col": row["col"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": is_correct
            })

            print(
                f"Cell ({row['row']},{row['col']}) | "
                f"GT: {true_label} | "
                f"Pred: {pred_label}"
            )

        tray_score = 100 * good_count / total_count

        print(f"Tray Quality Score: {tray_score:.2f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to: {OUTPUT_PATH}")

    # -------- SYSTEM METRICS --------
    y_true = results_df["true_label"].map(CLASS_TO_ID)
    y_pred = results_df["predicted_label"].map(CLASS_TO_ID)

    print("\nSystem-wide Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    print("\nSystem-wide Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    run_all_trays()
