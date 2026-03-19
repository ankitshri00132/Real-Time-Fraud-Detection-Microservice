from fastapi import FastAPI
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from src.inference.predictor import FraudPredictor
from src.monitoring.logger import get_logger


app = FastAPI()
logger = get_logger()

# Use an absolute path relative to this script for the model to prevent FileNotFoundError
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/fraud_pipeline_v01.pkl'))

predictor = FraudPredictor(model_path)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(transaction : dict):
    try:
        logger.info("Received Single Prediction Request")
        df = pd.DataFrame([transaction])

        prob,pred = predictor.predict(df)

        logger.info("Returning Prediction Response")
        return {
            "fraud_probability":float(prob[0]),
            "fraud_prediction":int(pred[0])
        }
    except Exception as e:
        logger.error(f"API Error : {str(e)}")
        return {"Error":"Prediction Failed"}
    

@app.post("/predict_batch")
def predict_batch(transactions : list[dict]):

    try:
        
        logger.info("Received Batch Prediction Request")
        df = pd.DataFrame(transactions)

        prob,pred = predictor.predict(df)

        logger.info("Returning Batch Prediction Response")
        return {
            "probablities":prob.tolist(),
            "predictions":pred.tolist()
        }
    
    except Exception as e:

        logger.error(f"API error : {str(e)}")
        return {"Error":"Prediction Failed"}