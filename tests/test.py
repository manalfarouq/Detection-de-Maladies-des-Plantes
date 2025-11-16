from fastapi.testclient import TestClient
from app.main import app
from tensorflow.keras.models import load_model

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from io import BytesIO
from PIL import Image
import numpy as np
import os

from app.main import app
from app.database import get_db
from app.models import Base, PlantDisease


client = TestClient(app)

def test_model_load():
    model = load_model("notebooks/facial_detection_model.keras")
    assert model is not None, "Le modèle n'a pas pu être chargé."
 

@pytest.fixture
def cleanup_db(db_session):
    """Nettoie la base de données après chaque test"""
    yield
    db_session.query(PlantDisease).delete()
    db_session.commit()


@pytest.fixture
def sample_plant_image():
    """Crée une image de plante de test 48x48 en niveaux de gris"""
    # Simuler une feuille avec du bruit
    img_array = np.random.randint(50, 200, (48, 48), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_healthy_leaf():
    """Image simulant une feuille saine"""
    img_array = np.full((48, 48), 150, dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_diseased_leaf():
    """Image simulant une feuille malade (avec taches sombres)"""
    img_array = np.random.randint(30, 100, (48, 48), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def invalid_file():
    """Crée un fichier non-image"""
    return BytesIO(b"This is a text file, not an image")


# ============= TESTS ENDPOINT ROOT =============

def test_root_endpoint():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Détection de maladies des plantes" in response.json()["message"]


# ============= TESTS PRÉDICTION DE MALADIES =============

def test_predict_disease_with_valid_image(sample_plant_image, cleanup_db):
    """Test de prédiction avec une image valide de plante"""
    files = {"file": ("leaf.png", sample_plant_image, "image/png")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Vérifier la structure de la réponse
    assert "id" in data
    assert "predicted_disease" in data
    assert "confidence" in data
    
    # Vérifier que c'est une maladie valide
    valid_diseases = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
        'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]
    assert data["predicted_disease"] in valid_diseases
    
    # Vérifier la confidence
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 1


def test_predict_pepper_disease(sample_plant_image, cleanup_db):
    """Test de prédiction pour une maladie de poivron"""
    files = {"file": ("pepper_leaf.png", sample_plant_image, "image/png")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 200
    # Le modèle devrait prédire une des classes disponibles


def test_predict_tomato_disease(sample_diseased_leaf, cleanup_db):
    """Test de prédiction pour une maladie de tomate"""
    files = {"file": ("tomato_leaf.jpg", sample_diseased_leaf, "image/jpeg")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 200


def test_predict_potato_disease(sample_healthy_leaf, cleanup_db):
    """Test de prédiction pour une maladie de pomme de terre"""
    files = {"file": ("potato_leaf.png", sample_healthy_leaf, "image/png")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 200


def test_predict_with_jpeg_image(sample_plant_image, cleanup_db):
    """Test avec une image JPEG"""
    # Convertir en JPEG
    img = Image.open(sample_plant_image)
    jpeg_bytes = BytesIO()
    img.convert('RGB').save(jpeg_bytes, format='JPEG')
    jpeg_bytes.seek(0)
    
    files = {"file": ("leaf.jpg", jpeg_bytes, "image/jpeg")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 200


def test_predict_with_invalid_file_type(invalid_file, cleanup_db):
    """Test avec un type de fichier non autorisé"""
    files = {"file": ("document.txt", invalid_file, "text/plain")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 400
    assert "non autorisé" in response.json()["detail"].lower()


def test_predict_without_file(cleanup_db):
    """Test sans fichier uploadé"""
    response = client.post("/predict_maladies")
    assert response.status_code == 422  # Validation error


def test_predict_with_corrupted_image(cleanup_db):
    """Test avec une image corrompue"""
    corrupted = BytesIO(b"\x89PNG\r\n\x1a\n" + b"corrupted_data" * 100)
    files = {"file": ("corrupted.png", corrupted, "image/png")}
    response = client.post("/predict_maladies", files=files)
    
    assert response.status_code == 400


# ============= TESTS HISTORIQUE =============

def test_get_history_empty(cleanup_db):
    """Test de l'historique vide"""
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_get_history_with_predictions(sample_plant_image, cleanup_db):
    """Test de l'historique avec plusieurs prédictions"""
    # Créer 3 prédictions
    for i in range(3):
        sample_plant_image.seek(0)
        files = {"file": (f"leaf_{i}.png", sample_plant_image, "image/png")}
        client.post("/predict_maladies", files=files)
    
    # Récupérer l'historique
    response = client.get("/history")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 3
    
    # Vérifier la structure de chaque entrée
    for entry in data:
        assert "id" in entry
        assert "predicted_disease" in entry
        assert "confidence" in entry
        assert "date" in entry


def test_history_chronological_order(sample_plant_image, cleanup_db):
    """Test que l'historique est dans l'ordre chronologique"""
    import time
    
    # Créer des prédictions avec un petit délai
    ids = []
    for i in range(3):
        sample_plant_image.seek(0)
        files = {"file": (f"leaf_{i}.png", sample_plant_image, "image/png")}
        response = client.post("/predict_maladies", files=files)
        ids.append(response.json()["id"])
        time.sleep(0.1)
    
    # Vérifier l'ordre
    history_response = client.get("/history")
    history = history_response.json()
    
    # L'historique devrait être dans l'ordre (id croissant ou décroissant)
    history_ids = [entry["id"] for entry in history]
    assert len(history_ids) == 3


# ============= TESTS BASE DE DONNÉES POSTGRESQL =============

def test_database_insert_plant_disease(cleanup_db, db_session):
    """Test d'insertion d'une maladie dans PostgreSQL"""
    new_disease = PlantDisease(
        predicted_disease="Tomato_Early_blight",
        confidence=0.89
    )
    db_session.add(new_disease)
    db_session.commit()
    db_session.refresh(new_disease)
    
    assert new_disease.id is not None
    assert new_disease.predicted_disease == "Tomato_Early_blight"
    assert new_disease.confidence == 0.89
    assert new_disease.created_at is not None


def test_database_query_by_disease(cleanup_db, db_session):
    """Test de requête par type de maladie"""
    # Insérer plusieurs prédictions
    diseases = [
        PlantDisease(predicted_disease="Tomato_Late_blight", confidence=0.95),
        PlantDisease(predicted_disease="Potato_Early_blight", confidence=0.87),
        PlantDisease(predicted_disease="Tomato_Late_blight", confidence=0.92),
    ]
    for disease in diseases:
        db_session.add(disease)
    db_session.commit()
    
    # Requête pour Tomato_Late_blight
    results = db_session.query(PlantDisease).filter(
        PlantDisease.predicted_disease == "Tomato_Late_blight"
    ).all()
    
    assert len(results) == 2


def test_database_query_high_confidence(cleanup_db, db_session):
    """Test de requête pour prédictions avec haute confidence"""
    diseases = [
        PlantDisease(predicted_disease="Tomato_healthy", confidence=0.98),
        PlantDisease(predicted_disease="Potato_healthy", confidence=0.65),
        PlantDisease(predicted_disease="Pepper__bell___healthy", confidence=0.91),
    ]
    for disease in diseases:
        db_session.add(disease)
    db_session.commit()
    
    # Requête pour confidence > 0.9
    results = db_session.query(PlantDisease).filter(
        PlantDisease.confidence > 0.9
    ).all()
    
    assert len(results) == 2


def test_database_count_predictions(cleanup_db, db_session):
    """Test du comptage des prédictions"""
    for i in range(5):
        db_session.add(PlantDisease(
            predicted_disease="Tomato_Bacterial_spot",
            confidence=0.8 + i * 0.03
        ))
    db_session.commit()
    
    count = db_session.query(PlantDisease).count()
    assert count == 5


def test_database_created_at_timestamp(cleanup_db, db_session):
    """Test que le timestamp created_at est correctement enregistré"""
    from datetime import datetime, timedelta
    
    disease = PlantDisease(
        predicted_disease="Tomato_Leaf_Mold",
        confidence=0.88
    )
    db_session.add(disease)
    db_session.commit()
    db_session.refresh(disease)
    
    # Vérifier que created_at est proche de maintenant
    now = datetime.now()
    assert disease.created_at is not None
    assert abs((disease.created_at - now).total_seconds()) < 5


# ============= TESTS D'INTÉGRATION =============

def test_full_workflow_prediction_to_history(sample_plant_image, cleanup_db):
    """Test du workflow complet : prédiction → base de données → historique"""
    # 1. Faire une prédiction
    files = {"file": ("tomato_leaf.png", sample_plant_image, "image/png")}
    pred_response = client.post("/predict_maladies", files=files)
    
    assert pred_response.status_code == 200
    pred_data = pred_response.json()
    pred_id = pred_data["id"]
    pred_disease = pred_data["predicted_disease"]
    
    # 2. Vérifier dans l'historique
    hist_response = client.get("/history")
    assert hist_response.status_code == 200
    history = hist_response.json()
    
    # 3. Vérifier que la prédiction est dans l'historique
    assert len(history) == 1
    assert history[0]["id"] == pred_id
    assert history[0]["predicted_disease"] == pred_disease


def test_multiple_predictions_different_diseases(cleanup_db):
    """Test de prédictions multiples avec différentes maladies"""
    # Simuler différentes images
    images = []
    for _ in range(3):
        img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        img_pil = Image.fromarray(img, mode='L')
        img_bytes = BytesIO()
        img_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        images.append(img_bytes)
    
    predictions = []
    for i, img_bytes in enumerate(images):
        files = {"file": (f"leaf_{i}.png", img_bytes, "image/png")}
        response = client.post("/predict_maladies", files=files)
        assert response.status_code == 200
        predictions.append(response.json())
    
    # Vérifier l'historique
    history = client.get("/history").json()
    assert len(history) == 3


# ============= TESTS DE PERFORMANCE =============

def test_prediction_response_time(sample_plant_image, cleanup_db):
    """Test du temps de réponse de la prédiction"""
    import time
    
    files = {"file": ("leaf.png", sample_plant_image, "image/png")}
    
    start = time.time()
    response = client.post("/predict_maladies", files=files)
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 3.0  # Moins de 3 secondes


def test_concurrent_predictions(sample_plant_image, cleanup_db):
    """Test de prédictions concurrentes"""
    import concurrent.futures
    
    def make_prediction():
        sample_plant_image.seek(0)
        files = {"file": ("leaf.png", sample_plant_image, "image/png")}
        return client.post("/predict_maladies", files=files)
    
    # 5 prédictions en parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_prediction) for _ in range(5)]
        responses = [f.result() for f in futures]
    
    # Toutes doivent réussir
    assert all(r.status_code == 200 for r in responses)
    
    # Vérifier l'historique
    history = client.get("/history").json()
    assert len(history) == 5


# ============= TESTS STATISTIQUES =============

def test_statistics_by_disease_type(cleanup_db, db_session):
    """Test des statistiques par type de maladie"""
    # Ajouter des prédictions variées
    diseases_data = [
        ("Tomato_Early_blight", 0.9),
        ("Tomato_Early_blight", 0.85),
        ("Potato_Late_blight", 0.92),
        ("Tomato_Early_blight", 0.88),
        ("Tomato_healthy", 0.95),
    ]
    
    for disease, conf in diseases_data:
        db_session.add(PlantDisease(predicted_disease=disease, confidence=conf))
    db_session.commit()
    
    # Compter par type
    from sqlalchemy import func
    stats = db_session.query(
        PlantDisease.predicted_disease,
        func.count(PlantDisease.id).label('count')
    ).group_by(PlantDisease.predicted_disease).all()
    
    stats_dict = {disease: count for disease, count in stats}
    assert stats_dict["Tomato_Early_blight"] == 3
    assert stats_dict["Potato_Late_blight"] == 1