import pandas as pd
from src.models.predict_pipeline import Predictor
from src.logger import logging
from src.exception import CustomException

def test_predict_single():
    import sklearn
    import sys
    print("\n--- TESTING ENVIRONMENT ---")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Python executable: {sys.executable}")
    print("-------------------------\n")
    try:
        logging.info("Testing model by Predicting sample test data.")
        pred = Predictor()
        sample = {
            "gender":"Female",
            "SeniorCitizen":0,
            "Partner":"Yes",
            "Dependents":"No",
            "tenure":10,
            "PhoneService":"Yes",
            "MultipleLines":"No",
            "InternetService":"DSL",
            "OnlineSecurity":"No",
            "OnlineBackup":"Yes",
            "DeviceProtection":"No",
            "TechSupport":"No",
            "StreamingTV":"No",
            "StreamingMovies":"No",
            "Contract":"Month-to-month",
            "PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 700.5
        }
        out = pred.predict_single(sample)
        assert "probability" in out and "label" in out
        assert 0.0 <= out["probability"] <= 1.0
        logging.info("Testing model Prediction successfully completed.")
    except Exception as e:
        raise CustomException(e)    