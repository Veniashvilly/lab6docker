import pandas as pd
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import joblib

df = pd.read_csv("https://raw.githubusercontent.com/Veniashvilly/dtset/refs/heads/main/Car_sales.csv")

df = df[[
    'Manufacturer', 'Model', 'Vehicle_type', 'Sales_in_thousands',
    'Engine_size', 'Horsepower', 'Curb_weight',
    'Fuel_efficiency', 'Price_in_thousands'
]].dropna().reset_index(drop=True)
cat_columns = ['Manufacturer', 'Model', 'Vehicle_type']
ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_columns] = ordinal.fit_transform(df[cat_columns])
X = df.drop(columns=['Price_in_thousands'])
y = df['Price_in_thousands']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
power = PowerTransformer()
y_scaled = power.fit_transform(y.values.reshape(-1, 1))
model = SGDRegressor(random_state=42)
model.fit(X_scaled, y_scaled.reshape(-1))

joblib.dump(model, "model.pkl")
joblib.dump(ordinal, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(power, "power_transformer.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
