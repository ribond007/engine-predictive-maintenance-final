import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

MODEL_REPO = "RishiBond/engine-predictive-maintenance-final"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model.pkl"
)

model = joblib.load(model_path)

def predict_engine(rpm, lub_oil_pressure, fuel_pressure,
                   coolant_pressure, lub_oil_temp, coolant_temp):

    features = np.array([[rpm, lub_oil_pressure, fuel_pressure,
                          coolant_pressure, lub_oil_temp, coolant_temp]])

    prediction = model.predict(features)[0]

    return "! Maintenance Required" if prediction == 1 else " Engine Normal"

demo = gr.Interface(
    fn=predict_engine,
    inputs=[gr.Number()]*6,
    outputs="text",
    title="Engine Predictive Maintenance System"
)

if __name__ == "__main__":
    demo.launch()

