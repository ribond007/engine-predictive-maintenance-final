from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score
from huggingface_hub import HfApi
import joblib
import os

HF_USERNAME = "RishiBond"

PROCESSED_DATASET = "engine-predictive-maintenance-processed"
MODEL_REPO = "predictive-maintenance-engine-model"  # MUST EXIST

def train_model():
    print("📥 Loading processed dataset...")
    dataset = load_dataset(f"{HF_USERNAME}/{PROCESSED_DATASET}")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    TARGET = "engine_condition"

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print("🚀 Training Random Forest with GridSearch...")
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10]
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
    preds = best_model.predict(X_test)

    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Best Params:", grid.best_params_)
    print("Recall:", recall)
    print("F1 Score:", f1)

    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.pkl"
    joblib.dump(best_model, model_path)

    print("☁️ Uploading model to Hugging Face...")
    api = HfApi()

    # IMPORTANT: repo already exists → no create_repo here
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pkl",
        repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
        repo_type="model"
    )

    print("✅ Model uploaded successfully.")

if __name__ == "__main__":
    train_model()
