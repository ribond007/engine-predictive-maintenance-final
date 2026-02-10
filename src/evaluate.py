import joblib
from datasets import load_dataset
from sklearn.metrics import classification_report

HF_USERNAME = "RishiBond"
PROCESSED_DATASET = "engine-predictive-maintenance-processed"

def evaluate():
    model = joblib.load("artifacts/model.pkl")
    test_ds = load_dataset(f"{HF_USERNAME}/{PROCESSED_DATASET}", split="test")
    df = test_ds.to_pandas()

    X = df.drop(columns=["engine_condition"])
    y = df["engine_condition"]

    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate()

