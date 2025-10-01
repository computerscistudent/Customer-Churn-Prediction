import joblib
from pathlib import Path
from src.models.predict_with_shap import predictor
from src.logger import logging
from src.exception import CustomException

model_path = Path("artifacts/best_model.joblib")
predictor_instance = predictor(model_path)


def get_predictor():
    try:
        logging.info("creating and returning the model instance so it only loads once.")
        return predictor_instance
    except Exception as e:
        raise CustomException(e)
