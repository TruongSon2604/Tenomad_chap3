from sqlalchemy import Column, Integer, String, Date,DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime as dat

class VehicleInfo(Base):
    __tablename__ = "vehicle_info"
    
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(20), unique=True, index=True)
    owner_name = Column(String(100), index=True)
    vehicle_brand = Column(String(50), index=True)
    model_type = Column(String(50), index=True)
    chassis_number = Column(String(50), unique=True, index=True)
    engine_number = Column(String(50), unique=True, index=True)
    production_year = Column(Integer, index=True)
    inspection_expiry  = Column(Date, index=True)
    created_at = Column(DateTime,default=dat.now, index=True)
