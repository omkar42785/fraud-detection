import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, scaler=None, training=True):

    X = df.drop("Class", axis=1)
    X = X.drop("Time", axis=1)
    if training:
        scaler = StandardScaler()
        X["Amount"] = scaler.fit_transform(X[["Amount"]])
    else:
        X["Amount"] = scaler.transform(X[["Amount"]])

    return X, scaler