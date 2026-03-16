from fastapi import FastAPI
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.inference.predictor import FraudPredictor

app = FastAPI()

# Use an absolute path relative to this script for the model to prevent FileNotFoundError
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/fraud_pipeline_v01.pkl'))

predictor = FraudPredictor(model_path)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(transaction : dict):
    df = pd.DataFrame([transaction])

    prob,pred = predictor.predict(df)

    return {
        "fraud_probability":float(prob[0]),
        "fraud_prediction":int(pred[0])
    }

@app.post("/predict_batch")
def predict_batch(transactions : list[dict]):
    df = pd.DataFrame(transactions)

    prob,pred = predictor.predict(df)

    return {
        "probablities":prob.tolist(),
        "predictions":pred.tolist()
    }