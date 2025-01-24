from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load models and vectorizer
rf = joblib.load("models/random_forest.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

app = FastAPI()

class SMSData(BaseModel):
    text: str

@app.post("/predict/")
def predict_spam(data: SMSData):
    features = vectorizer.extract_features([data.text])
    prediction = rf.predict(features)[0]
    return {"prediction": "spam" if prediction == 1 else "not spam"}
