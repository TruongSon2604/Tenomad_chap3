from pydantic import BaseModel
from datetime import datetime

class VehicleCreate(BaseModel):
    license_plate: str
    owner_name: str
    vehicle_brand: str
    model_type: str
    chassis_number: str
    engine_number: str
    production_year: int
    inspection_expiry: datetime

class VehicleResponse(VehicleCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

