from datasets import load_dataset
from huggingface_hub import HfApi
import os

DATASET_NAME = "engine-predictive-maintenance"
HF_USERNAME = "RishiBond"   
def register_dataset():
    dataset = load_dataset(
        "csv",
        data_files="data/raw/engine_data.csv"
    )

    dataset.push_to_hub(f"{HF_USERNAME}/{DATASET_NAME}")
    print("Dataset successfully pushed to Hugging Face Hub")

if __name__ == "__main__":
    register_dataset()

