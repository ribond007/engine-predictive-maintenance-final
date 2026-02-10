from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import pandas as pd
import os

# Hugging Face details
HF_USERNAME = "RishiBond"

RAW_DATASET_REPO = "engine-predictive-maintenance-data"
PROCESSED_DATASET_REPO = "engine-predictive-maintenance-processed"

def prepare_data():
    print("📥 Loading raw dataset from Hugging Face...")
    dataset = load_dataset(f"{HF_USERNAME}/{RAW_DATASET_REPO}", split="train")
    df = dataset.to_pandas()

    print("🧹 Basic data cleaning...")
    df = df.drop_duplicates()
    df = df.dropna()

    # ---- TARGET COLUMN CHECK ----
    # Change this ONLY if your target column name is different
    TARGET_COL = "engine_condition"

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("✂️ Splitting into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs("data/processed", exist_ok=True)

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("☁️ Uploading processed datasets to Hugging Face...")

    api = HfApi()
    api.create_repo(
        repo_id=f"{HF_USERNAME}/{PROCESSED_DATASET_REPO}",
        repo_type="dataset",
        exist_ok=True
    )

    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train.csv",
        repo_id=f"{HF_USERNAME}/{PROCESSED_DATASET_REPO}",
        repo_type="dataset"
    )

    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="test.csv",
        repo_id=f"{HF_USERNAME}/{PROCESSED_DATASET_REPO}",
        repo_type="dataset"
    )

    print("✅ Data preparation completed successfully.")

if __name__ == "__main__":
    prepare_data()
