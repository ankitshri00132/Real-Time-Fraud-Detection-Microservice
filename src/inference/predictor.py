import joblib
import numpy as np

class FraudPredictor:

    def __init__(self,model_path,threshold = 0.11):
        
        self.pipeline = joblib.load(model_path)
        self.threshold = threshold

    def predict(self,data):

        prob = self.pipeline.predict_proba(data)[:,1]
        pred = (prob>=self.threshold).astype(int)

        return prob, pred
    
    