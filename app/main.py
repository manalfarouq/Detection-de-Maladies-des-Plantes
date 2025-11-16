from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import shutil, uuid, os, cv2

from app.database import get_db, engine
from app.models import Base, PlantDisease
from app.predict import predict_disease

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Plant Detection API")


@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de Détection de maladies des plantes!"}


@app.post("/predict_maladies")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ALLOWED_TYPES = ["image/jpeg", "image/png"]
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Type non autorisé: {file.content_type}")

    # Génération d'un nom temporaire
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_ext}"

    # Sauvegarde temporaire
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()

    try:
        # Lecture de l'image
        img = cv2.imread(temp_filename)
        if img is None:
            raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

        # Prédiction
        confidence, predicted_disease = predict_disease(temp_filename)

        if confidence is None or predicted_disease is None:
            raise HTTPException(status_code=404, detail="Aucune maladie détectée.")

        # Sauvegarde en base
        db_pred = PlantDisease(
            predicted_disease=predicted_disease,
            confidence=float(confidence)
        )
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)

        # Réponse
        return {
            "id": db_pred.id,
            "predicted_disease": db_pred.predicted_disease,
            "confidence": round(db_pred.confidence, 2)
        }

    finally:
        # Suppression du fichier temporaire
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
            
@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    """
    Récupère toutes les prédictions enregistrées dans la base de données
    et les renvoie avec un affichage simple
    """
    predictions = db.query(PlantDisease).order_by(PlantDisease.created_at.desc()).all()
    result = []
    for p in predictions:
        result.append({
            "id": p.id,
            "predicted_disease": p.predicted_disease,
            "confidence": round(p.confidence, 2),  # arrondi à 2 décimales
            "date": p.created_at.strftime("%d/%m/%Y %H:%M:%S") if p.created_at else "N/A" 
        })

    return result               
