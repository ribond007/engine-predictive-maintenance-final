from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import pandas as pd

import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

HF_USERNAME = "RishiBond"
RAW_DATASET = "engine-predictive-maintenance"
PROCESSED_DATASET = "engine-predictive-maintenance-processed"

def prepare_data():
    # Load raw dataset from Hugging Face
    dataset = load_dataset(f"{HF_USERNAME}/{RAW_DATASET}", split="train")
    df = dataset.to_pandas()

    # Basic cleaning (drop duplicates if any)
    df = df.drop_duplicates()

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["engine_condition"]
    )

    # Convert back to HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Push to HF
    train_dataset.push_to_hub(f"{HF_USERNAME}/{PROCESSED_DATASET}", split="train")
    test_dataset.push_to_hub(f"{HF_USERNAME}/{PROCESSED_DATASET}", split="test")

    print("Train and test datasets uploaded successfully")

if __name__ == "__main__":
    prepare_data()

