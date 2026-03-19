import joblib
from src.monitoring.logger import get_logger
import numpy as np


logger = get_logger()


class FraudPredictor:

    def __init__(self,model_path,threshold = 0.11):
        
        self.pipeline = joblib.load(model_path)
        self.threshold = threshold

        logger.info("Model Loaded Successfully")
        

    def predict(self,data):

        try:
            logger.info("Prediction Started")

            prob = self.pipeline.predict_proba(data)[:,1]
            pred = (prob>=self.threshold).astype(int)   

            logger.info(f"Prediction Completed | Prob : {prob[0]} | Pred : {pred[0]}")

            return prob, pred
        
        except Exception as e:

            logger.error(f"Prediction Failed : {str(e)}")
            raise
    
    