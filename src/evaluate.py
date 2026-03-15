import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocessing import preprocess_data

# load model
model = joblib.load("models/fraud_model.pkl")

# load data
df = pd.read_csv("data/raw/creditcard.csv")

X, scaler = preprocess_data(df, training=False)
y = df["Class"]

# predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

print("\nROC AUC:", roc_auc_score(y, y_prob))