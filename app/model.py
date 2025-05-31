import joblib
import pandas as pd

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
power = joblib.load("power_transformer.pkl")
columns = joblib.load("columns.pkl")

def predict_price(data: dict) -> float:
    df = pd.DataFrame([data])
    cat_columns = ['Manufacturer', 'Model', 'Vehicle_type']
    df[cat_columns] = encoder.transform(df[cat_columns])
    df = df[columns]
    X_scaled = scaler.transform(df.values)
    y_scaled = model.predict(X_scaled)
    y_pred = power.inverse_transform(y_scaled.reshape(-1, 1))[0][0]

    return float(y_pred)
