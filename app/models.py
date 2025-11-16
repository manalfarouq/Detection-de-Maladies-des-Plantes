from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PlantDisease(Base):
    __tablename__ = "plant_diseases"

    id = Column(Integer, primary_key=True, index=True)
    predicted_disease = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())