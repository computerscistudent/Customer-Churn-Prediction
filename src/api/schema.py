from pydantic import BaseModel
from typing import Dict

class CustomerInput(BaseModel):
    gender: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str

class PredictionOutput(BaseModel):
    probability: float
    label: int
    top_features: Dict[str, float] = None