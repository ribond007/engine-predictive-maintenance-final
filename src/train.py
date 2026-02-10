from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score
from huggingface_hub import HfApi
import joblib
import pandas as pd
import os

HF_USERNAME = "RishiBond"
PROCESSED_DATASET = "engine-predictive-maintenance-processed"
MODEL_REPO = "engine-predictive-maintenance-model"

def train_model():
    # Load datasets
    train_ds = load_dataset(f"{HF_USERNAME}/{PROCESSED_DATASET}", split="train")
    test_ds = load_dataset(f"{HF_USERNAME}/{PROCESSED_DATASET}", split="test")

    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    X_train = train_df.drop(columns=["engine_condition"])
    y_train = train_df["engine_condition"]

    X_test = test_df.drop(columns=["engine_condition"])
    y_test = test_df["engine_condition"]

    # Model + GridSearch
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluation
    preds = best_model.predict(X_test)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Best Params:", grid.best_params_)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/model.pkl")

    # Push model to HF
    api = HfApi()
    api.upload_file(
        path_or_fileobj="artifacts/model.pkl",
        path_in_repo="model.pkl",
        repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
        repo_type="model"
    )

if __name__ == "__main__":
    train_model()

