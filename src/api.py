from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class Transaction(BaseModel):
    features: list



@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}



@app.post("/predict")
def predict(data: Transaction):

    features = np.array(data.features).reshape(1, -1)

    # scale the Amount column (last feature)
    features[:, -1] = scaler.transform(features[:, -1].reshape(-1,1)).flatten()

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction[0]),
        "fraud_probability": float(probability)
    }