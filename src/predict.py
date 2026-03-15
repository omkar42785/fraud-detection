import joblib
import pandas as pd

from preprocessing import preprocess_data


model = joblib.load("C:/Users/omkar/OneDrive/Desktop/projects/fraud-detection/models/fraud_model.pkl")
scaler = joblib.load("C:/Users/omkar/OneDrive/Desktop/projects/fraud-detection/models/scaler.pkl")

def detect_fraud(df):

    X, _ = preprocess_data(df, scaler=scaler, training=False)

    prob = model.predict_proba(X)

    normal_prob = prob[0][0]
    fraud_prob = prob[0][1]

    print(f"Normal probability: {normal_prob*100:.4f}%")
    print(f"Fraud probability: {fraud_prob*100:.4f}%")


  