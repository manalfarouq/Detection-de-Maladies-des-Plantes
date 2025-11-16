from pydantic import BaseModel
from datetime import datetime

class PlantPredictionCreate(BaseModel):
    predicted_disease: str
    confidence: float
    created_at: datetime
