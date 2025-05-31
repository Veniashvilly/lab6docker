from pydantic import BaseModel

class InputData(BaseModel):
    Manufacturer: str
    Model: str
    Vehicle_type: str
    Sales_in_thousands: float
    Engine_size: float
    Horsepower: float
    Curb_weight: float
    Fuel_efficiency: float
