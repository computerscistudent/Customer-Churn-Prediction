from fastapi import FastAPI, Depends, Form, Request
from src.api.schema import CustomerInput, PredictionOutput
from src.api.dependences import get_predictor
from fastapi.responses import RedirectResponse , HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Churn Prediction API")

@app.get("/" , response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionOutput)
def predict_customer(data: CustomerInput, predictor=Depends(get_predictor)):
    result = predictor.predict_single(data.dict())
    return PredictionOutput(**result)

@app.post("/explain", response_model=PredictionOutput)
def explain_customer(data: CustomerInput, predictor=Depends(get_predictor)):
    prediction = predictor.predict_single(data.dict())
    explanation = predictor.explain_single(data.dict())
    prediction["top_features"] = explanation
    return PredictionOutput(**prediction)


templates = Jinja2Templates(directory="templates")    

@app.post("/submit",response_class=HTMLResponse)
def submit_form(
    request: Request,
    gender: str = Form(...),
    tenure: int = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    Contract: str = Form(...),
    PaymentMethod: str = Form(...),
    InternetService: str = Form(...),
    predictor=Depends(get_predictor)
):
    data = {
        "gender": gender,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "InternetService": InternetService
    }
    result = predictor.predict_single(data)
    result["probability"] = round(result["probability"] * 100, 2)  # convert to %
    return templates.TemplateResponse("result.html", {"request": request, **result})
    
