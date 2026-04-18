import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------- CONFIG ----------------
CSV_PATH = Path("data/Annotations.csv")
IMAGE_DIR = Path("data/Images")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"


# ---------------- LOAD DATA ----------------
def load_annotations():
    df = pd.read_csv(CSV_PATH)

    required_cols = ["file", "classDescription"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["full_path"] = df["file"].apply(lambda x: str(IMAGE_DIR / x))
    df["exists"] = df["full_path"].apply(lambda x: Path(x).exists())

    missing = (~df["exists"]).sum()

    print(f"Total rows: {len(df)}")
    print(f"Missing images: {missing}")

    df = df[df["exists"]].copy()

    print("\nClass distribution:")
    print(df["classDescription"].value_counts())

    return df


# ---------------- SPLIT ----------------
def split_dataset(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["classDescription"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["classDescription"],
        random_state=42
    )

    return train_df, val_df, test_df


# ---------------- SAVE ----------------
def save_splits(train_df, val_df, test_df):
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("\nSaved split files:")
    print(TRAIN_CSV)
    print(VAL_CSV)
    print(TEST_CSV)

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


# ---------------- MAIN ----------------
def main():
    df = load_annotations()
    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
