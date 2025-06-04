from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services.car import VehicleService
from schemas.car_schema import VehicleCreate, VehicleResponse
from schemas.common import ResponseModel
from typing import List


router = APIRouter(prefix="/vehicles", tags=["vehicles"])

@router.post("/", response_model=VehicleResponse)
def create_vehicle(vehicle: VehicleCreate, db: Session = Depends(get_db)):
    service = VehicleService(db)
    return service.create_vehicle(vehicle)

@router.get("/{plate}", response_model=VehicleResponse)
def get_vehicle_by_plate(plate: str, db: Session = Depends(get_db)):
    service = VehicleService(db)
    vehicle = service.get_vehicle_by_plate(plate)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

@router.get("/", response_model=ResponseModel[List[VehicleResponse]])
def list_vehicles(db: Session = Depends(get_db)):
    service = VehicleService(db)
    vehicles = service.list_vehicles()
    return ResponseModel(
        status="success",
        message="List of vehicles",
        data=vehicles
    )

@router.delete("/{vehicle_id}", response_model=VehicleResponse)
def delete_vehicle(vehicle_id:int, db:Session = Depends(get_db)):
    service = VehicleService(db)
    vehicle = service.delete_vehicle(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

