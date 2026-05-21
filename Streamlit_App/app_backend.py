from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained pipeline
model = joblib.load("fake_job_model.pkl")

app = FastAPI(title="Fake Job Detection API")

# 🔹 Input schema
class JobInput(BaseModel):
    job_title: str
    description: str
    requirements: str
    benefits: str

# 🔹 Root endpoint
#@app.get("/")
#def home():
    #return {"message": "Fake Job Detection API is running"}

# 🔹 Prediction endpoint
@app.post("/predict")
def predict(job: JobInput):
    
    # ✅ Combine all fields into one text
    combined_text = f"""
    Title: {job.job_title}
    Description: {job.description}
    Requirements: {job.requirements}
    Benefits: {job.benefits}
    """
    
    # Predict
    prediction = model.predict([combined_text])[0]
    probability = model.predict_proba([combined_text])[0][1]
    
    # 🔥 Convert numeric → label
    label = "Fraudulent" if prediction == 1 else "Legitimate"
    
    return {
        "prediction": label,
        "probability_fraud": float(probability)
    }
    