import joblib
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException

MODEL_DIR = "artifacts"

class Predictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR,"baseline_model.joblib")
        obj = joblib.load(model_path)
        self.model = obj["model"]    

    try:
        def predict_single(self,row):
            logging.info("Predicting Single data points.")
            df = pd.DataFrame([row])

            if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
            prob = self.model.predict_proba(df)[:,1][0]
            label = int(prob>0.5)
            logging.info("Prediction of Single data points is successful.")
            return {"probability": float(prob), "label": label}  
    except Exception as e:
        raise CustomException(e)          
    
    def predict_singles(self,df:pd.DataFrame):
        try:
            logging.info("Predicting collective data points.")
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

            probs = self.model.predict_proba(df)[:,1]   
            labels = (probs>0.5).astype(int)
            logging.info("Prediction of collective data points is successful.")
            return pd.DataFrame({"probability": probs, "label": labels})
        except Exception as e:
            raise CustomException(e)