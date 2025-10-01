import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "gender": "Female",
    "tenure": 12,
    "MonthlyCharges": 70,
    "TotalCharges": 800,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "DSL"
}

response = requests.post(url, json=data)
print("Predict response:", response.json())


# Now test explain
url = "http://127.0.0.1:8000/explain"
response = requests.post(url, json=data)
print("Explain response:", response.json())
