from fastapi import FastAPI
from app.schema import InputData
from app.model import predict_price

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Car price prediction service is running"}

@app.post("/predict")
def predict(input_data: InputData):
    result = predict_price(input_data.dict())
    return {"predicted_price": result}
