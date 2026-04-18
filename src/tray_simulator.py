import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import math

CSV_PATH = "data/processed/test.csv"

OUTPUT_DIR = Path("assets/trays")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "tray_manifest.csv"

TRAY_ROWS = 3
TRAY_COLS = 3
CELL_SIZE = 224
PADDING = 20

TRAY_CAPACITY = TRAY_ROWS * TRAY_COLS


def build_all_trays():
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)

    num_images = len(df)
    num_trays = math.ceil(num_images / TRAY_CAPACITY)

    manifest_rows = []

    print(f"Total images: {num_images}")
    print(f"Generating {num_trays} trays")

    img_idx = 0

    for tray_num in range(num_trays):
        tray_h = TRAY_ROWS * CELL_SIZE + (TRAY_ROWS + 1) * PADDING
        tray_w = TRAY_COLS * CELL_SIZE + (TRAY_COLS + 1) * PADDING

        tray = np.ones((tray_h, tray_w, 3), dtype=np.uint8) * 255

        tray_id = f"tray_{tray_num+1:03d}"

        for r in range(TRAY_ROWS):
            for c in range(TRAY_COLS):
                if img_idx >= num_images:
                    break

                row = df.iloc[img_idx]

                img = cv2.imread(row["full_path"])
                img = cv2.resize(img, (CELL_SIZE, CELL_SIZE))

                y = PADDING + r * (CELL_SIZE + PADDING)
                x = PADDING + c * (CELL_SIZE + PADDING)

                tray[y:y+CELL_SIZE, x:x+CELL_SIZE] = img

                manifest_rows.append({
                    "tray_id": tray_id,
                    "row": r,
                    "col": c,
                    "x": x,
                    "y": y,
                    "file": row["file"],
                    "true_label": row["classDescription"]
                })

                img_idx += 1

        tray_path = OUTPUT_DIR / f"{tray_id}.jpg"
        cv2.imwrite(str(tray_path), tray)

        print(f"Saved {tray_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    print(f"\nSaved manifest: {MANIFEST_PATH}")
    print(f"Total records: {len(manifest_df)}")


if __name__ == "__main__":
    build_all_trays()

