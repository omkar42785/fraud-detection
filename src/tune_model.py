import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data

df = pd.read_csv("data/raw/creditcard.csv")

X, scaler = preprocess_data(df, training=True)
y = df["Class"]

param_dist = {
    "n_estimators": [100,200,300],
    "max_depth": [None,10,20],
    "min_samples_split": [2,5,10]
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=10,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

search.fit(X,y)

print("Best Parameters:", search.best_params_)