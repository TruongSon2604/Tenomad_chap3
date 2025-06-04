from sqlalchemy.orm import Session
from schemas.car_schema import VehicleCreate, VehicleResponse
from models.car import VehicleInfo
from repositories.car_repo import CarRepository

class VehicleService:
    def __init__(self, db: Session):
        self.repo = CarRepository(db)
    
    def create_vehicle(self, vehicle: VehicleCreate) -> VehicleInfo: 
        db_vehicle = self.repo.create_vehicle(vehicle)
        return db_vehicle
    
    def get_vehicle_by_plate(self, plate: str):
        return self.repo.get_vehicle_by_plate(plate)
    
    def list_vehicles(self):
        return self.repo.get_all()
    
    def delete_vehicle(self, vehicle_id: int):
        return self.repo.delete(vehicle_id)
    