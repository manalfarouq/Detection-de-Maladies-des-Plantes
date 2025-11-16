import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from io import BytesIO
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

from app.main import app
from app.database import get_db
from app.models import Base, PlantDisease

# --- DB test setup ---
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://user:password@localhost/test_plant_diseases"
)
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# --- Fixtures ---
@pytest.fixture(scope="session")
def model():
    return load_model("notebooks/facial_detection_model.keras")

@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def cleanup_db(db_session):
    yield
    db_session.query(PlantDisease).delete()
    db_session.commit()

@pytest.fixture
def sample_plant_image():
    arr = np.random.randint(50, 200, (48,48), dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return buf

@pytest.fixture
def sample_healthy_leaf():
    arr = np.full((48,48), 150, dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return buf

@pytest.fixture
def sample_diseased_leaf():
    arr = np.random.randint(30, 100, (48,48), dtype=np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return buf

@pytest.fixture
def invalid_file():
    return BytesIO(b"This is a text file, not an image")

# --- Tests ---
def test_model_load(model):
    assert model is not None

def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

def test_predict_valid(sample_plant_image, cleanup_db):
    files = {"file": ("leaf.png", sample_plant_image, "image/png")}
    r = client.post("/predict_maladies", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "id" in data and "predicted_disease" in data and "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_invalid_file(invalid_file, cleanup_db):
    files = {"file": ("doc.txt", invalid_file, "text/plain")}
    r = client.post("/predict_maladies", files=files)
    assert r.status_code == 400

def test_db_insert(db_session, cleanup_db):
    d = PlantDisease(predicted_disease="Tomato_Early_blight", confidence=0.9)
    db_session.add(d); db_session.commit(); db_session.refresh(d)
    assert d.id is not None
    assert d.confidence == 0.9
