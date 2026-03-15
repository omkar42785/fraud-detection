import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_data

# load dataset
df = pd.read_csv("data/raw/creditcard.csv")

X, scaler = preprocess_data(df, training=True)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained and saved")