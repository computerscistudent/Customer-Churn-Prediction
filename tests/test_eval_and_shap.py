# tests/test_eval_and_shap.py
import joblib
from src.models.predict_with_shap import predictor
from src.logger import logging
from src.exception import CustomException

def test_predict_and_explain_smoke():
    try:
        logging.info("Testing model by Predicting sample test data.")
        p = predictor()
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
        out = p.predict_single(sample)
        assert 0.0 <= out['probability'] <= 1.0
        expl = p.explain_single(sample)
        assert isinstance(expl, dict)
        logging.info("Testing model Prediction successfully completed.")
    except Exception as e:
        raise CustomException(e)