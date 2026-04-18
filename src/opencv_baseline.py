import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

TEST_CSV = "data/processed/test.csv"

CLASS_MAP = {
    "Defect_No": 0,
    "Defect_Shape": 1,
    "Defect_Object": 2,
    "Defect_Color": 3
}

CLASS_NAMES = [
    "Defect_No",
    "Defect_Shape",
    "Defect_Object",
    "Defect_Color"
]


# ---------------- SEGMENT BISCUIT ----------------
def segment_biscuit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)

    return thresh, largest


# ---------------- SHAPE FEATURE ----------------
def shape_score(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


# ---------------- COLOUR FEATURE ----------------
def colour_score(img, mask):
    mean_intensity = cv2.mean(img, mask=mask)[0]
    return mean_intensity


# ---------------- OBJECT DEFECT ----------------
def object_anomaly_score(mask):
    num_labels, labels = cv2.connectedComponents(mask)

    return num_labels


# ---------------- RULE-BASED CLASSIFIER ----------------
def classify(img):
    mask, contour = segment_biscuit(img)

    if contour is None:
        return 2

    circularity = shape_score(contour)
    intensity = colour_score(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        mask
    )
    objects = object_anomaly_score(mask)

    # ---- RULES ----
    if intensity < 90:
        return 3

    if circularity < 0.65:
        return 1

    if objects > 3:
        return 2

    return 0


# ---------------- MAIN EVALUATION ----------------
def main():
    df = pd.read_csv(TEST_CSV)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        img_path = row["full_path"]

        img = cv2.imread(img_path)

        pred = classify(img)

        true = CLASS_MAP[row["classDescription"]]

        y_true.append(true)
        y_pred.append(pred)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
