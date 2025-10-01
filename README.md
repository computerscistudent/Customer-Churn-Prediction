# Customer Churn Prediction API

This project is a machine learning application that predicts customer churn. It uses a trained model and exposes a REST API built with FastAPI to provide real-time predictions.

## Features

* Data preprocessing and feature engineering pipelines.
* Model training and evaluation.
* REST API for getting predictions.
* API endpoint to explain the features driving a prediction.
* Codebase structured for scalability and maintainability.

## Tech Stack

* **Language:** Python 3.12
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Web Framework:** FastAPI
* **Server:** Uvicorn

## Project Structure

A brief overview of the key directories in this project:

```
├── artifacts/      # Stores model files and other outputs
├── data/           # Contains raw and processed data
├── logs/           # For storing log files
├── notebooks/      # Jupyter notebooks for experimentation
├── src/            # Main source code for the application
│   ├── api/        # API related logic
│   ├── data/       # Data ingestion scripts
│   ├── features/   # Feature engineering scripts
│   └── models/     # Model training and prediction scripts
├── templates/      # HTML templates for the frontend
└── requirements.txt # Project dependencies
```

## Setup and Installation

Follow these steps to set up the project locally.

**1. Clone the repository:**
```bash
git clone [https://github.com/computerscistudent/Customer-Churn-Prediction.git](https://github.com/computerscistudent/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## How to Run

**1. Start the FastAPI application:**

Once the dependencies are installed, run the following command from the root directory:

```bash
uvicorn main:app --reload
```

The application will be running at `http://127.0.0.1:8000`.

**2. Access the API Documentation:**

You can view the interactive API documentation (provided by Swagger UI) by navigating to `http://127.0.0.1:8000/docs` in your browser.

## Example API Usage

You can send a `POST` request to the `/predict` endpoint with customer data to get a churn prediction.

