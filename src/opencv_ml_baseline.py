# src/opencv_ml_baseline.py

import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

TRAIN_CSV = "data/processed/train.csv"
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


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_intensity = gray.mean()
    std_intensity = gray.std()

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (
            4 * np.pi * area / perimeter**2
            if perimeter > 0 else 0
        )
    else:
        area = 0
        circularity = 0

    return [
        mean_intensity,
        std_intensity,
        edge_density,
        area,
        circularity
    ]


def build_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        img = cv2.imread(row["full_path"])

        features = extract_features(img)

        X.append(features)
        y.append(CLASS_MAP[row["classDescription"]])

    return np.array(X), np.array(y)


def main():
    X_train, y_train = build_dataset(TRAIN_CSV)
    X_test, y_test = build_dataset(TEST_CSV)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print(classification_report(
        y_test,
        preds,
        target_names=CLASS_NAMES
    ))

    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
