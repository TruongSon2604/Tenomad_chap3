from sqlalchemy.orm import Session
from models.car import VehicleInfo
from schemas.car_schema import VehicleCreate, VehicleResponse

class CarRepository:
    def __init__(self,db: Session):
        self.db = db
    
    def create_vehicle(self, vehicle:VehicleCreate) -> VehicleInfo:
        db_vehicle = VehicleInfo(**vehicle.dict())
        self.db.add(db_vehicle)
        self.db.commit()
        self.db.refresh(db_vehicle)
        return db_vehicle
    
    def get_vehicle_by_plate(self, plate: str):
        return self.db.query(VehicleInfo).filter(VehicleInfo.license_plate == plate).first()
    
    def get_all(self):
        return self.db.query(VehicleInfo).all()
    
    def delete(self, vehicle_id: int):
        vehicle = self.db.query(VehicleInfo).filter(VehicleInfo.id == vehicle_id).first()
        if vehicle:
            self.db.delete(vehicle)
            self.db.commit()
        return vehicle
         