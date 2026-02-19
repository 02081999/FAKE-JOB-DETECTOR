from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI
app = FastAPI(title="Fake Job Detection API")

# Define request schema
class JobPost(BaseModel):
    job_title: str
    description: str
    requirements: str
    benefits: str

# Endpoint to predict fake job
@app.post("/predict")
def predict_job(job: JobPost):
    # Combine text fields
    text = f"{job.job_title} {job.description} {job.requirements} {job.benefits}"
    
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    
    return {
        "prediction": "Fraudulent" if prediction == 1 else "Legitimate",
        "probability_fraud": float(probability) if probability is not None else "N/A"
    }
